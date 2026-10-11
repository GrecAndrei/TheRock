# Hardware and upstream follow-up, 2026-10-11

Host access is available following the initial restricted-session setup. Both
remote fetches completed; rocmtry main remains at `867ea43`, and llama.cpp master
is checked out separately at `966baae76b578d5d3dd4c179a75a35c80ff75577`.

The original benchmark process is absent from the now-visible host process
list. Its saved journal still has no exit record and no completed generation
result. Its historical exit reason remains unknown.

## Proven queue lifecycle

The old KFD probe reproduced `AMDKFD_IOC_CREATE_QUEUE: EINVAL` on the actual
RX 6650 XT. Evidence: `results/kfd-reproduce-VD9PNA/queue.log`.

Linux v6.18's `kfd_queue_buffer_get` requires queue pointers to name the start
page of their BO mapping and validates exact mapping sizes. The existing
dispatcher allocates its ring and index page as one BO; its write pointer names
a later page in that BO, and the ring's complete mapping is larger than the
declared ring. The driver also checks control-stack size and context-save
storage against device topology, while the original probe supplied zeroes.

The new `hardware_queue` probe uses existing Kyanite allocation and queue owners:

- separate 4096-byte ring and 4096-byte index BOs;
- an owned 4096-byte EOP buffer;
- topology-provided CWSR **11157504 bytes** and control stack **16384 bytes**;
- **32768 debugger bytes**, matching the gfx1032 single-XCC wave geometry;
- initialized save-area header and invalid ring packets;
- queue destruction before allocation cleanup, with process-exit retention on
  indeterminate queue/cleanup errors.

Three physical create/map/destroy/free repetitions passed, with exit status 0.
Evidence: `results/kfd-queue-UlEHQ0/queue.log`. This proves the chosen queue
arrangement, not kernel execution or model inference. The production
`KfdDirectDispatcher` has not yet been changed to own these auxiliary regions.

The probe compiled, passed Clippy with warnings denied, and passed its host
save-header layout test. Its linked libraries are libc and libgcc_s, with no
HIP/HSA userspace runtime. Kyanite commits:

- `e836679`: preserve imported bringup edits in the isolated engine branch.
- `329a56a`: add the current-driver queue lifecycle probe.

Driver reference: [Linux v6.18 kfd_queue.c](https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/amd/amdkfd/kfd_queue.c).
The save-header layout and debugger sizing were checked against the local
ROCR-Runtime `libhsakmt/src/queues.c` and `include/hsakmt/hsakmttypes.h`; these are
development references and are not linked by Kyanite.

## Native generation diagnostic

The preserved native source was built freshly with TheRock Clang/HIP 23, Release
configuration and two compiler jobs. Binary SHA256:
`6b451bd3babc7e8a83f5669841f120ded5194a597912c16518675a5203f5b928`.

A real chat prompt asks for an explanation of CPU/GPU cooperation for a 16 GB
model on an 8 GB GPU. Reasoning is displayed so its output remains in the raw
log. The request uses Q8 KV, 131072-slot capacity, 1.0 GiB requested target weight
budget, exact DFlash matrices, device checkpoints, 20 CPU workers and all new
tile/async experiments disabled. Its **101 physical prompt tokens** are not
128K history. There are no excluded warmups and no repeated-run certification.

Asset hashes:

- Target: `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
- Drafter: `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`.

Raw evidence is under `results/native-real-chat-wOMXgg`. The observed physical
prefill took **91.4 seconds**. At the first observed decode cycle the process
and system showed substantial memory pressure; system swap use was **8.8 GiB**.
This is a request diagnostic, not clean idle-host performance certification.
Its per-cycle displays must not be reported as sustained generation throughput.
The request hit its **360-second timeout**, and the managed command returned
**124**. The native process was subsequently absent. No complete answer or
sustained generation result was produced. The native child exit/signal was not
independently journaled: shell `pipefail` exited on the timeout before its
planned exit-file write. The timeout-wrapper status is recorded explicitly in
`records/native-chat-20261011-result.json`; it is not substituted for a native
child exit code. The raw log and input hashes are retained in `records`.

One observed cycle reported **75.5976 seconds** in feature injection. This is
phase attribution from the native log, not proof that the arithmetic itself took
that long: initialization, transfers and memory paging need separate diagnosis.
The visible output stopped during reasoning at `We need`. No per-cycle burst or
partial token count is promoted to a successful performance measurement.

## Next engineering gate

Integrate the proven mapping and auxiliary-region ownership into the production
dispatcher with failure/rollback tests. Then run bounded empty/write-index/
vector-add dispatch, completion, readback and teardown before implementing the
missing inference operators. Neither the queue probe nor a native chat result
can establish Kyanite's 10 tokens/s goal.
