#!/usr/bin/env python3
"""
Extract all unique GEMM shapes from a GGUF model file.

GGUF stores tensor metadata (name, shape, dtype) in the file header before
any weight data. We read only the header to enumerate every matrix multiply
the model will perform at inference time, without loading weights into memory.

Output: JSON list of {M, N, K, dtype, count, tensors} for each unique shape.
"""

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path


GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION_SUPPORTED = {2, 3}

# GGUF value types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types (quantization)
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
    34: "TQ1_0", 35: "TQ2_0",
}


class GGUFReader:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self._read_header()

    def _u8(self):  return struct.unpack("<B", self.f.read(1))[0]
    def _u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def _u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def _i32(self): return struct.unpack("<i", self.f.read(4))[0]
    def _i64(self): return struct.unpack("<q", self.f.read(8))[0]
    def _f32(self): return struct.unpack("<f", self.f.read(4))[0]
    def _f64(self): return struct.unpack("<d", self.f.read(8))[0]

    def _string(self):
        length = self._u64()
        return self.f.read(length).decode("utf-8", errors="replace")

    def _value(self, vtype):
        if vtype == GGUF_TYPE_UINT8:   return self._u8()
        if vtype == GGUF_TYPE_INT8:    return struct.unpack("<b", self.f.read(1))[0]
        if vtype == GGUF_TYPE_UINT16:  return struct.unpack("<H", self.f.read(2))[0]
        if vtype == GGUF_TYPE_INT16:   return struct.unpack("<h", self.f.read(2))[0]
        if vtype == GGUF_TYPE_UINT32:  return self._u32()
        if vtype == GGUF_TYPE_INT32:   return self._i32()
        if vtype == GGUF_TYPE_FLOAT32: return self._f32()
        if vtype == GGUF_TYPE_BOOL:    return bool(self._u8())
        if vtype == GGUF_TYPE_STRING:  return self._string()
        if vtype == GGUF_TYPE_UINT64:  return self._u64()
        if vtype == GGUF_TYPE_INT64:   return self._i64()
        if vtype == GGUF_TYPE_FLOAT64: return self._f64()
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self._u32()
            count = self._u64()
            return [self._value(elem_type) for _ in range(count)]
        raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _read_header(self):
        magic = self._u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic:#010x})")

        version = self._u32()
        if version not in GGUF_VERSION_SUPPORTED:
            raise ValueError(f"Unsupported GGUF version {version}")

        self.n_tensors = self._u64()
        n_kv = self._u64()

        # Read key-value metadata (we want arch info)
        self.metadata = {}
        for _ in range(n_kv):
            key = self._string()
            vtype = self._u32()
            self.metadata[key] = self._value(vtype)

        # Read tensor info
        self.tensors = []
        for _ in range(self.n_tensors):
            name = self._string()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            dtype = self._u32()
            # offset within data section (skip)
            self.f.read(8)
            self.tensors.append({"name": name, "dims": dims, "dtype": dtype})

    def close(self):
        self.f.close()


def is_weight_matrix(name: str, dims: list) -> bool:
    """Filter to tensors that participate in matrix multiplies."""
    if len(dims) < 2:
        return False
    # Skip embeddings, norms, biases, output logits (not batched GEMMs)
    skip = ("norm", "bias", "token_embd", "output.weight", "rope")
    if any(s in name for s in skip):
        return False
    return True


def gemm_shapes_for_tensor(name: str, dims: list, batch_sizes: list) -> list:
    """
    For a weight matrix of shape [N, K] (GGUF stores col-major),
    return the GEMM shapes (M, N, K) for each batch size M we care about.

    At inference: output = input @ weight.T
      input:  [M, K]
      weight: [N, K]  (stored transposed)
      output: [M, N]
    """
    if len(dims) == 2:
        K, N = dims[0], dims[1]
    else:
        # For higher-rank tensors flatten all but last dim
        K = dims[0]
        N = 1
        for d in dims[1:]:
            N *= d

    return [{"M": M, "N": N, "K": K} for M in batch_sizes]


def extract_shapes(model_path: Path, batch_sizes: list) -> dict:
    print(f"Reading {model_path.name}...", file=sys.stderr)
    reader = GGUFReader(model_path)

    arch = reader.metadata.get("general.architecture", "unknown")
    name = reader.metadata.get("general.name", model_path.stem)
    print(f"  Architecture: {arch}", file=sys.stderr)
    print(f"  Tensors: {reader.n_tensors}", file=sys.stderr)

    # Count unique GEMM shapes across all weight matrices
    shape_tensors = defaultdict(list)  # (M,N,K,dtype_name) -> [tensor_names]

    weight_tensors = [
        t for t in reader.tensors
        if is_weight_matrix(t["name"], t["dims"])
    ]

    print(f"  Weight matrices: {len(weight_tensors)}", file=sys.stderr)

    for tensor in weight_tensors:
        dtype_name = GGML_TYPE_NAMES.get(tensor["dtype"], f"type_{tensor['dtype']}")
        for shape in gemm_shapes_for_tensor(tensor["name"], tensor["dims"], batch_sizes):
            key = (shape["M"], shape["N"], shape["K"], dtype_name)
            shape_tensors[key].append(tensor["name"])

    reader.close()

    shapes = []
    for (M, N, K, dtype), tensor_names in sorted(shape_tensors.items()):
        shapes.append({
            "M": M, "N": N, "K": K,
            "dtype": dtype,
            "count": len(tensor_names),
            "tensors": tensor_names,
        })

    # Sort by compute intensity (M*N*K descending)
    shapes.sort(key=lambda s: s["M"] * s["N"] * s["K"], reverse=True)

    return {
        "model": name,
        "architecture": arch,
        "source": str(model_path),
        "batch_sizes": batch_sizes,
        "unique_shapes": len(shapes),
        "shapes": shapes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract GEMM shapes from a GGUF model for TensileLite tuning"
    )
    parser.add_argument("model", type=Path, help="Path to .gguf model file")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
        help="Token batch sizes to generate shapes for (default: 1 4 8 16 32)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write JSON output to file (default: stdout)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print human-readable summary instead of JSON"
    )
    args = parser.parse_args()

    result = extract_shapes(args.model, args.batch_sizes)

    if args.summary:
        print(f"\nModel: {result['model']}")
        print(f"Architecture: {result['architecture']}")
        print(f"Unique GEMM shapes: {result['unique_shapes']}")
        print(f"\nTop 10 shapes by compute (M×N×K):")
        print(f"{'M':>6} {'N':>6} {'K':>6}  {'dtype':<8}  {'count':>5}  example tensor")
        print("-" * 70)
        for s in result["shapes"][:10]:
            print(f"{s['M']:>6} {s['N']:>6} {s['K']:>6}  {s['dtype']:<8}  {s['count']:>5}  {s['tensors'][0]}")
    else:
        output = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(output)
            print(f"Written to {args.output}", file=sys.stderr)
        else:
            print(output)


if __name__ == "__main__":
    main()
