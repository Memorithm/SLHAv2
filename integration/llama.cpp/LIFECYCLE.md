# llama.cpp physical external-K lifecycle contract

This document describes the lifecycle surface supported by the pinned llama.cpp
reference integration (`b9860`, commit
`fdb1db877c526ec90f668eca1b858da5dba85560`). It is a correctness contract,
not a performance claim.

## Supported now

Physical `SLHA_EXTERNAL_K=1` supports the single-sequence autoregressive path
used by the real-model runner:

- one logical sequence and one KV stream;
- prompt prefill followed by sequential token append/decode;
- real cache positions supplied by llama.cpp `k_idxs`;
- full cache clear/reset, which also resets the external tile store;
- CCOS HOT/WARM residency under dense attention;
- quiescent CCOS COLD offload followed by exact HOT/WARM restoration before the
  next decode;
- EOS termination and ordinary context destruction/recreation.

The paired TinyStories real-model smoke exercises this supported surface on both
ordinary KV and physical external-K.

## Explicitly unsupported and fail-closed

The integration does **not** silently emulate the following operations:

- more than one logical sequence (`n_seq_max != 1`), even when llama.cpp uses a
  unified single physical KV stream;
- non-zero sequence position shifts (`seq_add`), because compressed RoPE K is
  not rotated after a logical shift;
- sequence position division (`seq_div`);
- llama.cpp state save/restore. The current llama state format knows only about
  llama-owned tensors; serializing the constant external-K sentinel and V cache
  without the physical SLHA tiles would create an incomplete state.

These paths are rejected before they can claim a valid external-K result.
Ordinary llama.cpp baseline mode is not restricted by this policy.

## Not yet claimed: sparse context reuse

`seq_rm`, `seq_keep`, same-stream `seq_cp`, SWA cell reuse, and arbitrary reuse
of holes inside the KV address space require a separate liveness design.

Today the external scorer uses a per-layer high-water mark and scores a dense
prefix of cache addresses. The CCOS ABI already exposes
`slha_elastic_cache_clear_slot`, but simply clearing a removed slot is not
sufficient: a dense score over a range containing that hole would then fail,
while retaining stale physical data is only safe as long as llama.cpp's KQ mask
continues to exclude that cell.

The next lifecycle increment must therefore synchronize llama cell liveness with
external-K scoring, not merely reclaim storage. Until that work has a dedicated
real-model trim/reuse test, sparse context reuse is intentionally not presented
as supported.

## State-format requirement for future persistence

State persistence may be enabled only when the serialized representation is
sufficient to reconstruct all semantics needed by external attention, including
at minimum:

- encoded K tile payloads;
- live/dead slot identity and stable cache addresses;
- codec/projection compatibility metadata;
- CCOS representation/tier information required for exact restoration, or a
  canonical representation from which residency can be rebuilt;
- validation that the restored model/projection geometry matches the state.

No partial state format is accepted merely to make the llama state API return
success.
