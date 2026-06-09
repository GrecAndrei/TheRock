#!/usr/bin/env python3
"""
Baseline GPU GEMM benchmark — no tuning, default ROCm behavior.

Runs each GEMM shape from extract_shapes.py using rocBLAS directly via
ctypes. Records TFLOPS and GB/s for each shape. This establishes the
pre-tuning baseline to compare against TensileLite-tuned kernels.

Requires: built ROCm (hip runtime + rocblas) in ROCM_PATH or TheRock build.
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import sys
import time
from pathlib import Path


ROCM_PATH = Path(os.environ.get(
    "ROCM_PATH",
    "/home/alex/TheRock/build/dist/rocm"
))


def find_lib(names: list) -> ctypes.CDLL:
    for name in names:
        for search in [ROCM_PATH / "lib", Path("/usr/lib"), Path("/usr/local/lib")]:
            p = search / name
            if p.exists():
                return ctypes.CDLL(str(p))
    raise RuntimeError(f"Could not find any of: {names}\nROCM_PATH={ROCM_PATH}")


def load_hip():
    lib = find_lib(["libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"])
    lib.hipMalloc.restype = ctypes.c_int
    lib.hipFree.restype = ctypes.c_int
    lib.hipMemcpy.restype = ctypes.c_int
    lib.hipDeviceSynchronize.restype = ctypes.c_int
    lib.hipGetErrorString.restype = ctypes.c_char_p
    lib.hipEventCreate.restype = ctypes.c_int
    lib.hipEventRecord.restype = ctypes.c_int
    lib.hipEventSynchronize.restype = ctypes.c_int
    lib.hipEventElapsedTime.restype = ctypes.c_int
    lib.hipEventDestroy.restype = ctypes.c_int
    return lib


def load_rocblas():
    lib = find_lib(["librocblas.so", "librocblas.so.4", "librocblas.so.3"])
    lib.rocblas_create_handle.restype = ctypes.c_int
    lib.rocblas_destroy_handle.restype = ctypes.c_int
    lib.rocblas_sgemm.restype = ctypes.c_int
    lib.rocblas_hgemm.restype = ctypes.c_int
    return lib


def check(status, name, hip):
    if status != 0:
        msg = hip.hipGetErrorString(status)
        raise RuntimeError(f"{name} failed ({status}): {msg}")


WARMUP_RUNS = 3
BENCH_RUNS  = 10


def bench_sgemm(hip, blas, handle, M: int, N: int, K: int) -> dict:
    """Benchmark fp32 SGEMM for shape (M, N, K)."""
    import ctypes

    sz_a = M * K * 4  # float32
    sz_b = K * N * 4
    sz_c = M * N * 4

    d_a = ctypes.c_void_p()
    d_b = ctypes.c_void_p()
    d_c = ctypes.c_void_p()

    check(hip.hipMalloc(ctypes.byref(d_a), sz_a), "hipMalloc A", hip)
    check(hip.hipMalloc(ctypes.byref(d_b), sz_b), "hipMalloc B", hip)
    check(hip.hipMalloc(ctypes.byref(d_c), sz_c), "hipMalloc C", hip)

    alpha = ctypes.c_float(1.0)
    beta  = ctypes.c_float(0.0)

    ROCBLAS_OPERATION_NONE      = 111
    ROCBLAS_OPERATION_TRANSPOSE = 112

    def run():
        return blas.rocblas_sgemm(
            handle,
            ROCBLAS_OPERATION_NONE,      # transA
            ROCBLAS_OPERATION_TRANSPOSE, # transB (weight matrix is N,K stored)
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K),
            ctypes.byref(alpha),
            d_a, ctypes.c_int(M),
            d_b, ctypes.c_int(N),
            ctypes.byref(beta),
            d_c, ctypes.c_int(M),
        )

    # Warmup
    for _ in range(WARMUP_RUNS):
        check(run(), "rocblas_sgemm warmup", hip)
    hip.hipDeviceSynchronize()

    # Timed runs using HIP events
    ev_start = ctypes.c_void_p()
    ev_stop  = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(ev_start))
    hip.hipEventCreate(ctypes.byref(ev_stop))

    hip.hipEventRecord(ev_start, None)
    for _ in range(BENCH_RUNS):
        run()
    hip.hipEventRecord(ev_stop, None)
    hip.hipEventSynchronize(ev_stop)

    ms = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(ms), ev_start, ev_stop)
    hip.hipEventDestroy(ev_start)
    hip.hipEventDestroy(ev_stop)

    hip.hipFree(d_a)
    hip.hipFree(d_b)
    hip.hipFree(d_c)

    avg_ms = ms.value / BENCH_RUNS
    flops  = 2 * M * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12

    # Memory: read A(M*K) + B(N*K) + write C(M*N), all fp32
    bytes_rw = (M*K + N*K + M*N) * 4
    bandwidth_gbs = bytes_rw / (avg_ms * 1e-3) / 1e9

    return {
        "avg_ms": round(avg_ms, 4),
        "tflops": round(tflops, 4),
        "bandwidth_gbs": round(bandwidth_gbs, 2),
    }


def get_device_info(hip) -> dict:
    # hipGetDeviceProperties is complex to call via ctypes — use rocminfo instead
    import subprocess, re
    rocminfo = ROCM_PATH / "bin" / "rocminfo"
    env = {**os.environ, "LD_LIBRARY_PATH": str(ROCM_PATH / "lib")}
    try:
        out = subprocess.check_output(
            [str(rocminfo)], stderr=subprocess.DEVNULL, text=True, env=env
        )
        # rocminfo lists CPU agents first, then GPU agents. Split by agent block
        # and return info from the first block whose Name: field contains "gfx".
        for agent in out.split("*******"):
            gfx = name = vram = ""
            for line in agent.splitlines():
                line = line.strip()
                if line.startswith("Name:") and "gfx" in line and not gfx:
                    raw = line.split(":", 1)[1].strip()
                    m = re.search(r"gfx\d+", raw)
                    gfx = m.group(0) if m else raw
                if "Marketing Name" in line:
                    name = line.split(":", 1)[1].strip()
                if "Memory" in line and "Size" in line:
                    vram = line.split(":", 1)[1].strip()
            if gfx:
                return {"name": name, "gfx": gfx, "vram": vram}
        return {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Baseline GEMM benchmark using default ROCm/rocBLAS"
    )
    parser.add_argument(
        "shapes_json", type=Path,
        help="JSON from extract_shapes.py (or - for stdin)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write JSON results to file"
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Only benchmark top N shapes by compute (default: all)"
    )
    args = parser.parse_args()

    if str(args.shapes_json) == "-":
        shape_data = json.load(sys.stdin)
    else:
        shape_data = json.loads(args.shapes_json.read_text())

    print(f"Loading HIP and rocBLAS from {ROCM_PATH}...", file=sys.stderr)
    hip  = load_hip()
    blas = load_rocblas()

    handle = ctypes.c_void_p()
    blas.rocblas_create_handle(ctypes.byref(handle))

    device = get_device_info(hip)
    print(f"GPU: {device.get('name', 'unknown')} ({device.get('gfx', '?')})", file=sys.stderr)

    shapes = shape_data["shapes"]
    if args.top:
        shapes = shapes[:args.top]

    print(f"\nBenchmarking {len(shapes)} shapes (fp32 SGEMM baseline)...", file=sys.stderr)
    print(f"{'M':>6} {'N':>6} {'K':>6}  {'ms':>8}  {'TFLOPS':>8}  {'GB/s':>8}", file=sys.stderr)
    print("-" * 55, file=sys.stderr)

    results = []
    for shape in shapes:
        M, N, K = shape["M"], shape["N"], shape["K"]
        try:
            perf = bench_sgemm(hip, blas, handle, M, N, K)
            results.append({**shape, **perf})
            print(
                f"{M:>6} {N:>6} {K:>6}  "
                f"{perf['avg_ms']:>8.3f}  "
                f"{perf['tflops']:>8.4f}  "
                f"{perf['bandwidth_gbs']:>8.2f}",
                file=sys.stderr
            )
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6}  ERROR: {e}", file=sys.stderr)
            results.append({**shape, "error": str(e)})

    blas.rocblas_destroy_handle(handle)

    output = {
        "model": shape_data["model"],
        "model_arch": shape_data.get("architecture", "unknown"),
        "device": device,
        "rocm_path": str(ROCM_PATH),
        "tuned": False,
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "results": results,
    }

    out_json = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(out_json)
        print(f"\nWritten to {args.output}", file=sys.stderr)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
