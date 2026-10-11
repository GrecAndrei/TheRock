# Research shortlist for gfx1032, checked 2026-10-11

These are hypotheses, not proven ports. Rank reflects our own measured engine
bottlenecks, not the papers' headline speedups. No paper was installed or linked
into Kyanite, and no result below demonstrates 10 tokens/s on this machine.

## 1. Tensor-level placement and exposed transfer cost

[ATSInfer, July 2026 preprint, v2](https://arxiv.org/html/2607.10183v2)
profiles individual tensors, plans residency under a memory budget, and chooses
dynamic transfers using execution costs and available overlap. It separates
small activation transfers from large weight movement. Its evaluation uses
NVIDIA RTX platforms, not gfx1032; CUDA copy-engine assumptions need independent
AMD validation. Relevant sections: 4.2–4.4 and 5.1.

Local hypothesis: replace the fixed residency priority and per-format row splits
with measured marginal latency saved per resident byte. Start with FFN gate/up/
down and output projections. Measure CPU compute, H2D bytes, exposed copy waits
and GPU compute separately; don't use overlapping phase labels as a roofline.
Preserve the existing operations and enforce buffer lifetimes and exact outputs.
Whole-model pinning is not imported: our own bounded staging already regressed.

## 2. Smaller speculative verification, when it actually saves work

[Dovetail, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.879/)
places drafting on the GPU and verification on the CPU, and finds that reducing
candidate count can improve speed despite lower acceptance length. Its trained
draft architecture and LLaMA/Vicuna experiments are not drop-in replacements for
our supplied Qwen/DFlash2 checkpoints. Read sections 3, 4 and the hardware table.

[Official DFlash implementation](https://github.com/z-lab/dflash) currently
recommends blocks no larger than five specifically for quantized **MLX** kernels;
this is not an AMD recommendation. The
[DFlash paper's block-size ablation](https://arxiv.org/html/2602.06036v2)
also discusses the drafting-quality/verification-cost trade-off.

Local hypothesis: compare real verification widths 2, 4 and 8 while retaining
greedy target confirmation and every committed output ID. Merely passing fewer
valid positions to our hard-wired batch-eight projection kernels does **not**
remove their padded arithmetic; a genuine variable-width path is needed before
claiming this optimization. Measure tokens/cycle, acceptance by position and
total wall time, not acceptance percentage alone.

## 3. Wide projections plus chunk-parallel recurrent prefill

[Gated Delta Networks, ICLR 2025](https://arxiv.org/html/2412.06464v3)
derives a chunkwise formulation of the gated delta recurrence using WY-style
transforms and matrix operations. The newer
[Qwen FlashQLA implementation](https://github.com/QwenLM/FlashQLA)
provides fused chunked-prefill kernels, with releases through September 2026.
It requires recent NVIDIA architectures and CUDA; its Tensor Core kernels are
not usable directly on the RX 6650 XT. See the paper's section 3 and the
implementation's requirements and forward API.

Local hypothesis: first implement genuine 32/64-token projection tiles that
reuse each quantized weight block, rather than looping over eight-token units.
Then evaluate chunkwise recurrent execution against the existing serial FP32
state. All 48 recurrent layers, convolution history, gates, final state and five
DFlash taps must stay correct, including tails and nonzero initial state.
Algebraic equivalence does not imply bit-exact floating-point equivalence;
existing correctness assertions are not relaxed to make a candidate pass.

Hardware caveat: AMD introduces WMMA in
[its RDNA 3 guide](https://gpuopen.com/learn/wmma_on_rdna3/). Our gfx1032 path must
use its own legal SIMD/LDS implementation, not promise NVIDIA/WMMA speedups.

## 4. Different plans for prefill and decode

[FlexInfer, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/698cfaf72a208aef2e78bcac55b74328-Paper-Conference.pdf)
selects execution policies separately for the two phases using profiled compute
and transfer costs. Its evaluated server processors and acceleration features
do not establish performance on our desktop CPU. Relevant sections: 4–6.

Local hypothesis: use wide GPU-oriented prompt processing, but choose decode
CPU/GPU shards from actual batch width and tensor type. Prefill acceleration
reduces time to first token; it does not, by itself, establish 10 decode tokens/s.

## Outside the unchanged correctness contract

[SpecPrefill, ICML 2025](https://arxiv.org/abs/2502.02789)
selects a subset of prompt tokens for target processing. That cannot certify
our required physically processed 131072-token prompt.
[HGCA, July 2025](https://arxiv.org/html/2507.03153v1)
uses sparse approximations on older CPU-resident KV entries; its lossless merge
does not make the preceding sparsification exact. Keep both out of the default
path. New Gated DeltaNet-2 models likewise cannot replace this checkpoint's
recurrence merely because their architecture is newer.

## Concrete next gates

1. Add per-projection compute/transfer attribution on diagnostic runs only;
   measure residency candidates before changing the default placement policy.
2. Check DFlash2 acceptance against an independent reference implementation:
   low observed acceptance is not proof of a draft bug. Audit feature taps,
   positions, convolution anchor and selector, with the same quantized assets.
3. Prototype one real wide-prefill projection, preserving quantization and
   accumulation order. Gate exact component/state/output parity before timings.
4. Run independent real prompts and repeated complete requests for promotion;
   finish with the unchanged physical-128K certification protocol.

Reference repository HEADs observed with `git ls-remote` (not cloned or ported):
DFlash `07ebd93db9f472af339b644bb70221ad8428328a`;
FlashQLA `da06429d54b0f577de0a638f451ac8f0b395e0ac`.
Web documentation was read from its current branch; any code adaptation must
first pin and inspect the actual source and license at a specific commit.
