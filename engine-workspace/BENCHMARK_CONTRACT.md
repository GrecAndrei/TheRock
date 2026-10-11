# Honest generation evidence

The goal is >=10 real, committed target tokens/s on the RX 6650 XT. No throughput
result is produced by workspace setup or host tests.

- Preserve the official target/drafter model bytes and quantization; record
  SHA256 for both before inference evidence is accepted. Freeze source, local
  diff, executable hash, compiler flags and runtime settings for every run.
- Record actual device identity, driver/kernel, VRAM/RAM, clock/power state,
  CPU workers, placement, cache precision, context occupancy and prompt IDs.
- Time the complete generation interval with a monotonic wall clock and ensure
  device work has completed at its endpoint. Include draft, target verification,
  rollback/commit, sampling and transfer overhead. Count only committed target
  outputs. Draft proposals, rejected tokens, internal batches and synthetic
  zero-filled history are never generated-output throughput.
- Record model load, prefill, TTFT, timed decode, acceptance, output token IDs and
  raw output separately. Report prompt-inclusive request speed as a separate
  metric; a decode result cannot stand in for responsiveness or TTFT.
- Keep practical short/medium prompts as explicitly labelled workloads. Context
  capacity is never reported as populated history. Preserve the existing final
  goal: three independent **physical 131072-token prefills**, eight excluded
  warmup outputs and 256 timed committed outputs per run, one greedy stream,
  unchanged frozen corpus and canonical tokenizer. Report each TPS, arithmetic
  mean, median and dispersion. Final success requires mean >=10.0 with the
  correctness gates satisfied; short-prompt success does not certify 128K.
- Establish independent tokenizer, quantization and model numerical parity;
  cover causal attention, recurrent state, prefill boundaries, every speculative
  accepted prefix, rollback/commit and session cleanup. Internal self-consistency
  alone is insufficient. Document tolerances and any token divergence.
- Comparisons use the same workload, model bytes, sampling and declared quality
  policy. Different quantization or placement is a separately labelled result.
  A llama.cpp default is only a candidate until it wins on this device.
- Save exact commands, complete raw stdout/stderr and explicit child exit status.
  Missing exit records or partial generation never pass. Preserve failed runs;
  create new result directories rather than overwriting prior evidence.
- Generation measurements require an idle, isolated workload. CPU correctness
  tests and modeled planner numbers cannot support an inference speed claim.

The copied historical probe is evidence of memory bringup and queue failure,
not kernel correctness. The old native baseline remains incomplete; this
contract does not retroactively certify it.
