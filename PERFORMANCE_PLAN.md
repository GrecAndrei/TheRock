# Native 128K performance design and baseline

This is the design record for the pure-TheRock Qwen3.8-27B/DFlash2 engine.
It covers only the standalone HIP executable in this directory. It must not
link to, invoke, or copy code from an external inference runtime.

## Target and measurement contract

- **Target:** `Qwen3.8-27B-UD-Q4_K_M.gguf` with
  `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` on the local 16-CU Radeon RX 6650 XT
  (`gfx1032`, 7.98 GiB) and i5-14600K.
- **Required final benchmark:** one warmed interactive chat stream with an
  *actual tokenized* 131,072-token context, greedy sampling, and 256 generated
  target tokens. Report the arithmetic mean decode rate of three clean timed
  runs (goal: at least 10.0 generated target tokens/s), each individual rate,
  DFlash acceptance statistics, and TTFT separately.
- **Correctness rule:** a speculative proposal is never a generated token.
  For deterministic greedy mode, every committed token must match the native
  target-only oracle. Rejected target and draft state must be restored before
  the next cycle.

## Baseline collected before implementation

All commands below use only TheRock source and the model files as data inputs.
`./build.sh` completed with `ninja: no work to do`.

```text
$ build/qwen38-native device
AMD Radeon RX 6650 XT; gfx1032; 7.98 GiB; 16 CUs; wave32

$ build/qwen38-native plan-128k TARGET DFLASH
Q8 target KV:        4.25 GiB
N23KV4 target KV:    2.25 GiB
DFlash2 weights:     1.06 GiB
room for target weights with N23KV4: 4.09 GiB

$ build/qwen38-native bench-kv4 131072 5 64
single-query K score:           2.133 ms/layer, 212.40 GB/s
8-query K score:                6.933 ms/layer, 65.34 GB/s
8-query fused K+V softmax:     25.634 ms/layer, max error 2.33e-09

$ build/qwen38-native target-verify-batch TARGET 131072 0 3 6.39
blocks: 0.727 s, 0.731 s, 0.723 s
rate: 11.005 candidate tokens/s
resident target cache: 6.39 GiB (179 matrices)

$ build/qwen38-native dflash-verify TARGET DFLASH 131072 0 5.0
proposal acceptance: 0/7
initial target: 1.186 s; draft: 0.130 s; verifier: 1.987 s

$ build/qwen38-native mtp-test TARGET 131072 0 3 6.39
three scalar target steps: about 1.07 s/step
```

These results are **not** an end-to-end success claim:

1. The N23KV4 benchmark uses synthetic K/V data. It establishes the numerical
   kernel result and an isolated long-history bandwidth measurement, not model
   decode quality.
2. `target-verify-batch` currently calls `simulate_full_attention_window()` but
   the active `AttentionDeviceState` is constructed with a hard-coded 4096
   entries in `NativeTargetSession`. It reports a 131072 capacity but does not
   retain/read a real 128K model history. Its 11.005 figure is candidate
   verification rate, not accepted generated-token throughput.
3. The cold DFlash check accepted no draft tokens. The MTP test prints an MTP
   prediction but does not compare it to the correctly advanced next target
   token, so it is not an acceptance test.
4. The native interactive loop currently advances recurrent/convolution state
   for all eight verifier positions but `rewind()` only adjusts the attention
   count. It also does not roll back the DFlash injected cache. Therefore a
   rejection corrupts later speculative state and current chat throughput must
   not be presented as quality-qualified generation.

## Correctness work after the baseline

The baseline caveats above describe the pre-fix implementation. The native
engine now has a distinct long-context path with the following evidence:

- `AttentionDeviceState` allocates an authoritative, device-resident **Q8_0
  key and value cache** when the requested capacity exceeds 4096. For 131072
  tokens and 16 full-attention layers this is 4.25 GiB; no cache entry is
  shifted or dropped before the requested capacity is reached.
- The long path uses partitioned online-softmax attention with a logical
  per-query causal length, rather than the old dynamic-shared-memory score
  buffer. Its cache stride is the declared capacity, while its reduction range
  is the active causal prefix; this prevents both a false 4096 limit and
  cross-batch future-token attention.
- The short-cache scalar oracle and long Q8 pipeline agree for the
  deterministic eight-token fixture:

  ```text
  $ build/qwen38-native target-compare-batch TARGET 131072 0 1.0
  reference: 108518 108518 26100 15639 56836 56836 56836 292
  pipeline:  108518 108518 26100 15639 56836 56836 56836 292
  result: match
  ```

- The same Q8 long path passes `target-rollback-test ... 131072 0 1.0`, so a
  rejected speculative suffix cannot change the replayed target sequence.
  `dflash-commit-test ... 131072 0 1.0` also passes with a three-token target
  state and a matching three-entry DFlash cache.
- The pipeline now captures all five DFlash target-feature snapshots for each
  valid verifier item. The chat scheduler injects exactly the committed
  anchor/accepted-input prefix after replay and checks that the injected cache
  length advances exactly once per committed target token.

These are correctness-foundation results only. They do **not** establish
quality at an active 131072-token prompt, nonzero DFlash acceptance, or the
required end-to-end throughput.

## Paper-to-implementation mapping

### DFlash: *Block Diffusion for Flash Speculative Decoding* (Sections 3--4,
Appendix A.3)

| Paper mechanism | Native seam | Required implementation/validation |
|---|---|---|
| A block diffusion drafter proposes all block positions in parallel and the target verifies them together. | `NativeDflashEncoder::propose`, `NativeTargetSession::step_batch8_pipelined`, and the chat loop in `src/main.cpp`. | Preserve the checkpoint's eight-position block (one anchor + seven proposals); do not claim speed from a larger untrained block. Commit only the target-confirmed prefix plus the target bonus token. |
| Fuse five target representations and inject them as persistent K/V in every draft layer. | `NativeTargetSession::dflash_features`, `NativeDflashEncoder::encode` and `inject`. | The model metadata says target layers `[6,20,34,48,62]`. Capture the output of layers `[5,19,33,47,61]`, which is the corresponding *input* to those target layers, then prove it with target/DFlash fixture output. Keep all five injected K/V histories transactionally aligned with target state. |
| Draft quality, not merely draft speed, determines the accepted length. | Selector lattice in `NativeDflashEncoder::select`; diagnostic `dflash-verify*` commands. | Multi-cycle greedy equivalence now passes and acceptance is nonzero, but acceptance remains prompt dependent and full selector parity is still required. Approximate draft kernels remain opt-in. |
| DFlash's target context is refreshed per target token, while injected K/V is reused. | Batched target feature capture plus `NativeDflashEncoder::inject_batch`. | Implemented for committed tokens only. Rollback and cache-length fixtures pass; authoritative feature-value parity remains pending. |
| The target verifier dominates only if proposal/verification transfer boundaries are small. | `DevicePipeline`, hybrid projection paths, final norm/head path. | Hidden state and projection intermediates remain device-side; compact tokens and required feature snapshots cross the boundary. Output-head and dependency-aware scheduling are still being tuned. |

### DeepSeek-V4: *Towards Highly Efficient Million-Token Context Intelligence*
(Sections 2.3 and 3.5)

DeepSeek-V4's CSA/HCA weights and sparse-indexing architecture cannot be
retrofitted into Qwen3.8 without changing the model and violating output-quality
constraints. The applicable lessons are implementation patterns, not permission
to replace Qwen attention with untrained compression or sparsity.

| Paper lesson | Safe native application | Prohibited shortcut |
|---|---|---|
| Separate bounded current state from long historical KV, with cache layout co-designed for kernels. | Implement one authoritative 128K target cache in native Q8_0 blocks, explicitly reporting capacity, active length, locations and numerical error. | Dropping old KV, sparse selecting arbitrary history, or substituting learned CSA/HCA entries without Qwen-compatible trained weights. |
| Use block-aligned layout and kernels for heterogeneous cache policies. | Align 32-value Q8 blocks and partitioned eight-query verifier tiles; avoid the old full 4096-entry D2D shift. Preserve MRoPE/global position across the full context. | Treating `CONTEXT=131072` as evidence when only a 4096-entry sliding cache is active. |
| Long-context throughput is a cache-management and kernel-scheduling problem, not solely a matvec problem. | Record resident-weight hits/misses, PCIe copy bytes/wait time, cache location, per-layer attention time, and verifier cycle timing. Use those data to choose residency/staging size. | Reporting isolated matrix or K-only benchmarks as chat throughput. |
| Multi-token prediction can reduce serial decoding when the target validates it. | Validate the supplied `blk.64` MTP head against the next correctly advanced target token, then use it only through the same transaction/verification contract as DFlash. | Accepting MTP output directly or comparing it to the wrong target position. |

## Ordered implementation gates

1. **Transactional target/speculative state — implemented and fixture-tested.**
   Snapshots cover recurrent matrices, convolution history, target attention
   metadata, and DFlash injected state. Captured-prefix commit avoids replaying
   a rejected suffix through 64 layers.
2. **Real 128K-capacity state — implemented; semantic occupancy still pending.**
   The authoritative path uses device Q8_0 K/V with separate capacity and active
   strides, causal per-query lengths, and preserved MRoPE/global position. A
   physical full scan is measured, but no actually tokenized 131072-token model
   fixture has completed.
3. **Native DFlash greedy equivalence — implemented for current fixtures.**
   Feature injection is committed-token-only, acceptance is nonzero, and the
   16-token speculative fixture matches sequential target output. Independent
   feature/selector parity and broader prompts remain required. N23 draft and
   lower-bit target KV experiments remain outside the exact default path.
4. **Optimize the measured critical path — active.** Exact AVX2 GEMM8,
   CPU/GPU row sharding, CPU recurrent-gate overlap, and barrier-free Q8
   attention are present. Target weight movement and full-window verification
   remain too slow.
5. **Final benchmark harness/result — pending.** It must build a true
   131072-token chat prompt, warm separately, run three 256-token greedy trials,
   retain raw records, and reject any rate based on proposal tokens or fake
   context occupancy.

## Source seams to inspect first

- `src/main.cpp`: `NativeTargetSession::step_batch8_pipelined`, `rewind`,
  `NativeDflashEncoder`, and `chat`.
- `src/kernels.cpp`: `DevicePipeline`, `AttentionDeviceState`,
  `pipeline_attention_batch8`, and N23KV4 kernels.
- `src/kernels.h`: testable/cache-safe interfaces needed to express snapshot,
  restore, telemetry, and true-context attention.

The pre-existing local edits to `src/main.cpp`, `src/kernels.cpp`, and
`src/kernels.h` are part of the starting point and must be integrated rather
than discarded.

## Correctness and hot-path update

A later authoritative GGUF-format audit found two target-critical layout bugs:

- Qwen3.5 GGUF conversion tiles recurrent V heads as `[replica][K head]` and
  applies the permutation to V/Z/A/B/conv/out tensors. The matching K head is
  therefore `v_head % 16`, not a contiguous `v_head / 3` group.
- Q6_K's second packed 64-byte low-quant stripe represents activation values
  128--255. The old kernel advanced the activation by 64 and consequently
  duplicated values 64--191 while omitting 192--255. Single and batch kernels
  plus their independent host fixture now use distinct packed and value
  offsets.

After these fixes, the xhigh chat fixture produces the coherent deterministic
reply `Hello! How can I help you today?`, and DFlash cycles accept as many as
7/7 proposals. The native Q8 long path still matches the scalar token fixture,
and captured-prefix recurrent commit passes the rollback fixture.

The rejection path no longer runs a second 64-layer target pass. During the
verifier it retains only projected QKV, normalized K/V and decay/beta factors
(about 35 MiB), then restores the checkpoint and reapplies the accepted
recurrent updates. Attention KV, logits and five target-feature snapshots are
already causal and are committed by shortening their logical length. This
changed rejected cycles from about 4.43 s to the same one-pass cost as fully
accepted cycles.

Measured optimization progression at short active history but declared 131072
capacity (`RESIDENT_GIB=1.0`):

```text
before correctness qualification: target verifier about 2.20 s/block
three-way FFN transfer staging:   target verifier 1.31 s/block
N23Q4 draft-only residency:       proposal 0.18 -> 0.11 s/cycle
captured-prefix commit:           rejection 4.43 -> 1.36 s/cycle
qualified Hello chat:             1.50 -> 3.57 generated tok/s
```

The Q8 long-attention kernel now groups eight query/head pairs per workgroup and
loads each exact Q8 K/V vector once for those queries. Experimental CPU/GPU row
splitting is exposed only through `QWEN_NATIVE_HYBRID*`; on this machine it
measured roughly 1.26--1.29 s per target block and did not materially beat the
1.31 s staged GPU path. It is not the default.

The remaining limit is weight movement. The official target is 15.33 GiB,
whereas Q8 target KV alone occupies 4.25 GiB on the 7.98-GiB card. Host-to-device
copy bandwidth measures about 13.4 GB/s. With checkpoint width fixed at eight,
the target must either execute a large fraction of streamed work concurrently
on the CPU or obtain an equally quality-preserving reduction in movement. N23
target KV is not such evidence: it changed greedy fixtures and remains an
experiment, not the correctness path.

A matrix-level CPU/GPU schedule now computes the recurrent decay/beta projection
on a persistent 24-worker AVX2 pool while HIP computes QKV. Unlike early generic
row splitting, this was useful: the default short-window verifier improved from
1.31 to 1.22 s/block and still passes sequential-Q8 versus batch-Q8 fixtures.
The `speculative-equivalence` fixture independently generates the target-only
stream one committed Q8 token at a time, then runs DFlash prefix commit from the
same prompt. Its 16-token fixture matched exactly (7 cycles, 10/49 accepted
draft proposals).

The first full-window timing command advanced the zero-filled, physically
allocated Q8 cache to 131064 entries and measured an actual 131072-entry scan
without claiming semantic prompt occupancy:

```text
$ build/qwen38-native target-long-window-bench TARGET 131072 248068 1.0
active context: 131072/131072
total: 1.637 s; recurrent 0.529; attention 0.159; FFN 0.889; head 0.016
```

HIP phase timings are asynchronous, so part of long attention is charged to the
following FFN interval; total block time is authoritative. At that checkpoint,
even perfect 7/7 acceptance yielded only `8 / (1.637 + 0.11) = 4.58` generated
tokens/s. This was an executor ceiling at that implementation checkpoint, not a
proof of a hardware limit: profiling subsequently found repeated Q8 cache reads
and scalar CPU quant unpacking that could still be removed.

## Current implementation checkpoint

The sections above preserve the chronological baseline and correctness history.
The current source has since added the following exact or target-verified paths.

### Exact CPU GEMM8 and heterogeneous row execution

`src/cpu_matvec.cpp` and `src/cpu_matvec.h` implement reusable structure-of-arrays
packs and AVX2/FMA/F16C batch-eight decoders for Q4_K, Q5_K, Q6_K, and IQ4_XS.
The persistent 24-worker pool decodes each original GGUF row once and accumulates
eight activations. `bench-matvec` now reports resident GPU, staged GPU, scalar
CPU, and SoA CPU measurements from the same real tensor.

Representative steady measurements are:

| Format | SoA throughput | Typical CPU/GPU maximum delta |
|---|---:|---:|
| Q4_K | 22.4 GB/s | order `1e-5` |
| Q5_K | 23.9 GB/s | order `1e-5` |
| Q6_K | 18.9 GB/s | order `1e-5` |
| IQ4_XS | about 19 GB/s | order `1e-5` to `3e-5` |

The target tensor byte mix is approximately Q5_K 29.4%, IQ4_XS 28.9%, Q4_K
25.6%, and Q6_K 11.0%, so these four kernels cover about 95.9% of the checkpoint.
`QWEN_NATIVE_HYBRID=1` assigns disjoint output-row ranges to CPU and GPU. The
format-specific split can be overridden by `QWEN_NATIVE_SOA_SPLIT`; fixed split
searches and the separate `QWEN_NATIVE_HYBRID_FFN` fused experiment did not
produce a decisive improvement and remain opt-in.

The hybrid output-head experiment is also opt-in. One short verifier run
improved, but a 2.5 GiB residency attempt failed with:

```text
hipMalloc native matvec weight staging: out of memory
```

No output-head result obtained by exceeding the available VRAM is considered a
valid configuration.

### Exact Q8 long-attention launch search

The original grouped exact-Q8 kernel issued six independent query groups for
each KV head and therefore could reread the physical 4.25 GiB cache multiple
times. New launch modes assign one wave32 per query while clustering 8, 12, 16,
or 24 queries in a workgroup. They preserve FP32 ordered online-softmax, exact
Q8 dequantization, per-query causal limits, and token-level target comparison.
The barrier-free 16-wave form (`QWEN_NATIVE_Q8_TILED=warp16`) was best on the
local RDNA2 card; 24 waves reduced workgroup occupancy.

A partition sweep selected 64 partitions at 131072 entries. Representative
rewound synthetic measurements were:

```text
warp8:   best total about 1.155 s/block
warp12:  pass 3 1.156 s (attention 0.410 s)
warp16:  pass 3 1.132 s (attention 0.398 s)
warp24:  pass 3 1.200 s (attention 0.452 s)
```

The `warp16` path also passed `target-compare-batch` at 131072 capacity. These
runs use a physically allocated and fully scanned, zero-filled Q8 history. They
are timing evidence only and must not be described as a tokenized 128K prompt.

### DFlash semantic and residency update

An audit against the official Transformers/DFlash model definitions confirmed
and implemented:

- mask token `248070`, eight-position block, selector top-k 16;
- target feature outputs after zero-based layers `[5, 19, 33, 47, 61]`;
- full 128-dimensional Qwen3 RoPE in the drafter;
- a symmetric noncausal 2048-token attention window;
- the checkpoint's `xhigh` reasoning system turn and system token ID `8678`;
- batched committed-token feature encoding/injection;
- no rejected target or draft suffix in persistent state.

The default DFlash preload now keeps the official quantized draft matrices
resident instead of converting them to N23. It reports about 2.05 GiB of total
resident target-plus-draft storage across 53 matrices with a 1.0 GiB target
budget. Proposal latency is normally about 0.10--0.12 s. N23 draft execution is
available only through `QWEN_NATIVE_DRAFT_N23=1` and is not used for exact-path
claims.

Prompt-dependent acceptance has included `0/7`, `1/7`, `5/7`, and `7/7` cycles.
For the coherent `Hello` fixture, one observed five-cycle pattern was
`1, 7, 0, 5, 7`, producing `Hello! How can I help you today?`. Selector
diagnostics found rejected target tokens at ranks 1 and 6 in two cycles, but
also outside top 16 in another cycle. This motivates an exact adaptive tree
verifier, but no tree path is currently committed.

A second unchanged DFlash block was also evaluated without claiming it as an
official larger block. When the first block was fully accepted, a stale second
block failed to predict the required target bonus token in all three observed
full-accept cycles. Therefore simple two-block chaining is rejected. The
`QWEN_NATIVE_CASCADE_TRACE` variable retains this diagnostic; it is not an
execution mode or speed result.

### Latest end-to-end observations

The current recommended local command is:

```bash
QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_Q8_TILED=warp16 \
  build/qwen38-native chat TARGET DFLASH 131072 1.0
```

Recent short-history target batches have taken roughly 0.80--0.90 s after
first-use packing. A representative trace was:

```text
block 2: 0.873 s
block 3: 0.825 s
block 4: 0.815 s
```

Before native IQ3_S replaced the quality-changing Q8 conversion, coherent chat
runs measured about 5.31--5.77 generated tokens/s. The current direct-IQ3_S
path measured 5.55 generated tokens/s on the same short `Hello` fixture.
Proposal count is not included in these values. First-use
packing and prefill are reported separately and have varied substantially with
host cache/system state.

A mapped-host direct GPU experiment did not remove the movement limit: Q4_K
measured 6.43 GB/s when the kernel read registered host memory directly, versus
10.19 GB/s for staged H2D plus kernel on the same tensor. Direct mapped weights
are therefore rejected for the hot path.

Lossless compression samples across the dominant target quant formats saved
only 0--2% and sometimes expanded, so lossless checkpoint compression is not a
viable residency solution.

## Current completion status

Completed foundations:

- pure-TheRock build/runtime and direct GGUF execution;
- native 131072-capacity Q8 KV state and streaming chat;
- transactional target/DFlash prefix commit and rollback;
- sequential-Q8 versus batch-Q8 token checks;
- 16-token target-only versus speculative greedy equivalence;
- corrected major quant, recurrent layout, RoPE, mask, selector, and DFlash
  feature-lifecycle bugs;
- exact CPU/GPU heterogeneous experiments and physical full-window timing.

Still required before any final success claim:

1. Independent hot-path numerical tests/oracles for Q5_K, IQ4_NL, IQ4_XS,
   attention, convolution, recurrence, FFN, and logits. Native IQ3_S now has
   a real-tensor scalar-CPU versus batch-GPU check.
2. Authoritative DFlash parity for dynamic convolution, feature alignment,
   selector/cache protocol, and long-window behavior.
3. Unicode tokenizer regex/pretokenization validation.
4. A target cycle fast enough that measured acceptance can sustain at least
   10 generated tokens/s; the current approximately 1.13 s synthetic full
   window plus draft cost cannot do so even at perfect width-eight acceptance.
5. The actual tokenized 131072-token chat benchmark: warm-up excluded, one
   deterministic greedy stream, 256 generated target tokens, three clean timed
   runs, each TTFT/rate/acceptance record, and an arithmetic mean at or above
   10.0 generated tokens/s.
6. Final dependency/diff review and documented DeepSeek-V4/DFlash technique
   effects.

The performance goal is therefore **not met**. The next exact optimization
route is a dependency-aware CPU/GPU matrix schedule, further Q8 cache reuse,
and—only if deterministic target equivalence can be maintained—an adaptive
DFlash verification DAG. Changing the checkpoint, quantizing the target/KV
more aggressively, lowering context/output size, changing hardware, or counting
draft proposals are not acceptable completion routes.

## Latest bounded scheduler and benchmark work

A bounded implementation pass followed the dependency-aware scheduler review:

- FFN gate/up now share one normalized-activation download and one CPU pool job,
  with GPU shards enqueued back-to-back and a join only before SiLU.
- Full-attention Q/K/V have a grouped exact hybrid path with one activation
  download and one CPU job. Resident matrices stay on the normal GPU path;
  nonresident matrices use disjoint CPU/GPU output rows and join before
  attention.
- The CPU pool remains persistent with capacity 24, but defaults to 20 active
  workers while HIP is concurrent. `QWEN_NATIVE_CPU_THREADS` permits bounded
  tuning; oversubscribing 24 helped isolated CPU tests but often slowed the
  full executor.
- An 8 MiB double-buffered H2D/kernel row pipeline was implemented behind
  `QWEN_NATIVE_GPU_PIPELINE`. Sweeps from 2 to 256 MiB were neutral or slower
  (`~0.83--0.95 s` short blocks), so the direct shard transfer remains default.
- A 16-query LDS Q8 attention mode (`warp16lds`) loads K/V once per workgroup
  tile and uses two barriers per 16 tokens. It preserved target token fixtures
  but measured about 0.418--0.430 s for full-window attention versus about
  0.398--0.407 s for barrier-free `warp16`, so it remains a rejected comparison
  mode.
- Pre-unpacking Q4_K/Q5_K block scale metadata into a slightly larger SoA cache
  reduced Q4_K from roughly 25 to 20 GB/s and was reverted. The original-byte-
  sized SoA pack remains selected.

After grouped Q/K/V, a representative short test still measured about
`0.814--0.832 s` per target block, with attention about `0.061--0.062 s`; the change is
correct but not material to the 10 tok/s gap. A subsequent physical full-window
pass remained about `1.15--1.19 s`, dominated by roughly `0.40 s` attention and
`0.45--0.49 s` FFN. `target-compare-batch` and the 16-token
`speculative-equivalence` fixture continue to match after these changes.

The adversarial scheduler review sets the exact kill conditions:

1. Width eight cannot reach 10 tok/s unless full-window target verification is
   at most about `0.69 s` even with perfect acceptance and current draft cost.
2. For a measured committed length `c = 256 / cycles`, the exact condition is
   `target/cycle <= c/10 - draft/cycle - injection/cycle`. Around five committed
   tokens/cycle, the target budget is below `0.39 s`.
3. Any target-only/speculative mismatch, lower-precision target/KV substitute,
   IQ3_S quality shortcut, or DFlash selector/cache divergence kills the route.

### Reproducible benchmark harness

`benchmark-128k` is now implemented with a fixed contract:

```text
benchmark-128k TARGET DFLASH OUTPUT.jsonl [RESIDENT_GIB] [RUNS]
```

It constructs an exactly 131072-token ChatML prompt from tokenizer-produced
paragraph segments, preserves the complete token vector in
`OUTPUT.jsonl.prompt.tokens`, allocates room for prompt plus continuation,
physically prefills every target Q8 KV entry and committed DFlash feature, then
excludes an eight-generated-token warm-up. Each timed run commits exactly 256
greedy target-confirmed tokens. JSONL records include model paths/sizes,
hardware, environment controls, hashes and raw output token IDs, initialization,
TTFT, total/phase decode times, cycles, proposal/acceptance counts, residency,
and arithmetic mean. Runs fail on early EOS, inter-run token mismatch, incomplete
prompt occupancy, cache misalignment, or mean below 10.0 tok/s.

A smoke start verified:

```text
prompt tokens: 131072
prompt FNV64:  0x4158e53b26398dc3
mode:          greedy, one stream, block 8
output target: 256 committed tokens
```

The smoke was intentionally terminated before expensive prefill. The complete
three-run record is still absent; at approximately 0.8 s per batch-eight
prefill block, one 128K prefill is multi-hour. It would be misleading to spend
three such runs and present their expected sub-10 result as goal completion
before the executor crosses the scheduler kill threshold.

### Native IQ3_S correction

The four target IQ3_S FFN tensors no longer undergo the earlier one-time Q8_0
runtime requantization. A native scalar and batch-eight GPU decoder now consumes
the original 110-byte/256-value block directly, including the canonical
512-entry D4 lattice, two high-index bits per eight values, separate sign bits,
and odd per-32-value scales. On real `blk.11.ffn_gate.weight` data, the
independent scalar CPU and GPU outputs agree within `8.11e-6`; the staged GPU
path measured 4.045 ms and 9.47 GB/s.

This correction is not merely cosmetic: deterministic verifier fixtures changed
from the old converted path (for example, later batch predictions changed from
`256, 220, ...` to `248069, 271, ...`). Sequential native target and batch
pipeline outputs still match, and 16-token speculative equivalence still
passes. The source GGUF bytes are unchanged and there is no target IQ3_S
requantization in the active path. First-pass startup also avoids materializing
four expanded Q8 matrices. Steady full-window time remains about 1.18 s, so the
quality fix does not clear the throughput blocker.
