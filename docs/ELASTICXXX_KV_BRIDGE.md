# SLHAv2 ↔ ElasticXxx KV transaction bridge v1

Status: **optional real-consumer integration; no performance or quality claim**.

`slhav2-vram` owns the physical SLHA tile cache and codec semantics. ElasticXxx
owns the generic `OBSERVE → PLAN → VALIDATE → ACT → VERIFY → COMMIT/ROLLBACK`
transaction. The optional Cargo feature `elasticxxx` depends only on the public
`elastic` facade and binds the bridge to exact ElasticXxx revision
`36e1b73022d6bf3f0bc4cd6ae40ee1bc95e468b6`.

## Physical transition

The v1 bridge exposes the physical cache's own logical resource identity and
resident-budget headroom as source-bound ElasticXxx capacity observation evidence.
That observation flows through the ElasticXxx V2 current-state forecast boundary
(zero horizon, no calibrated confidence claim) before Boolean eligibility.

The v1 physical actuator deliberately supports one narrow transition only:

```text
SLHAv2 HOT tile (128 resident bytes)
        ↓ Reencode@representation
SLHAv2 WARM tile (96 resident bytes + 32-byte reversible residual backing)
```

HOT/WARM are represented as distinct SLHA representation contracts while the
ElasticXxx `KvResidency` remains the explicit runtime-defined
`slhav2-host-cache` class. The bridge does **not** relabel COLD/offload as a
representation transition.

SLHAv2's WARM transition keeps enough backing to reconstruct the exact original
HOT tile. `PreserveContents` is reported true only because the adapter verifies
that reconstruction byte-for-byte. `PreserveIdentity` is bound to the cache's
monotonic slot generation rather than the recyclable physical slot index.

## Fail-closed authority boundary

`SlhaKvTransitionBackendV1` implements ElasticXxx `KvTransitionBackendV1` and:

- reuses the exact resource identity owned by `ElasticKvCache`;
- observes current `free_bytes()` through the same shared cache handle;
- derives page identity from SLHAv2's monotonic slot generation;
- rejects non-HOT sources, missing/recycled slots, codec drift and byte drift;
- checks the exact source under the same mutex that performs HOT→WARM;
- verifies WARM residual backing and exact HOT reconstruction after mutation;
- performs immediate physical rollback if the atomic mutation cannot establish
  the target state;
- supports the generic ElasticXxx rollback path WARM→HOT;
- rejects external `UpholdContract` invariants it does not implement.

The BE14d Boolean capacity preflight remains planning evidence only. Tests cover
fresh sufficient (`True`) and explicit insufficient (`False`) evidence from the
real physical cache observer plus an explicit unavailable (`Unknown`) provider.
The `True` path records `current-state` forecast metadata before it reaches the
existing ElasticXxx structural validator and transactional runtime.

## Scope limits

This contract does not claim:

- GPU-device residency (the current `ElasticKvCache` physical bytes are host
  `Vec<u8>` storage);
- COLD/offload integration;
- arbitrary SLHA codec-to-codec re-encoding;
- quality equivalence or model-level accuracy;
- a speedup or memory-product claim.

Those require separate versioned contracts and evidence. The SLHAv2 in-repo
Elastic incubator remains untouched; this bridge consumes the external
ElasticXxx contract only when the `elasticxxx` feature is enabled.
