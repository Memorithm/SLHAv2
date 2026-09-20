# ElasticXxx ELANG8a consumer contract — SLHAv2 KV

Status: cross-repository consumer qualification.

Source/target boundary for this revision:

- ElasticXxx source: `26cbcdc73cfd08121593cf1a034720ee93816fe9`;
- SLHAv2 target baseline: `4a7cbeb871430301b101dc4ee4291b6db2a84d53`.

SLHAv2 remains the owner of tile layout, codecs, HOT/WARM/COLD residency,
physical cache mutation and cache-specific verification. ElasticXxx remains the
owner of generic observation/admission/transaction contracts.

## Package identity convergence

The optional Cargo dependency now consumes the current package identity
`memorithm-elastic` while retaining the local Rust alias `elasticxxx`. The git
dependency is pinned to the exact source commit above. A unit test checks both
the package name and revision string so a later dependency update cannot leave
the runtime evidence constant silently stale.

## Existing physical contract retained

The V1 bridge is preserved:

- source-bound free-capacity evidence;
- `False`/`Unknown` capacity evidence stops before mutation;
- exact logical page identity uses the monotonic slot generation rather than
  recyclable slot index;
- current source bytes/codec/tier are rechecked under the same mutex boundary;
- ElasticXxx trusted transaction validation precedes physical HOT→WARM packing;
- verification checks the exact physical WARM bytes/descriptor;
- rollback restores the exact original HOT bytes.

ELANG8a does not replace these semantics.

## V2 fixed-width representation/precision composition

The additive V2 bridge exposes ElasticXxx's
`RepresentationPrecisionKvBindingV1` only when SLHAv2 has an honest fixed-width
precision declaration.

Today that means **uniform signed INT4 only**:

```text
codec = int4
precision = 4 declared bits/scalar
target = exact slhav2.int4.warm representation/schema/epoch
mechanism = Reencode
```

A real BE14e `screen_with_trace` report must select that candidate. The report's
DecisionTrace, source representation, target representation/schema/epoch and
mechanism are then composed with the exact physical `KvTransitionPlan` before
the transition can be treated as one coherent planning artifact.

A precision floor above four bits rejects the INT4 candidate before this
composition succeeds.

## Deliberately unsupported fixed-width mappings

NF4, MIXED, TQ3 and MIX3 remain `KvPrecision::Custom` in the current SLHAv2
bridge. Their storage/quantization semantics are not represented by one honest
fixed scalar bit width. Therefore:

- `fixed_width_precision_candidate()` returns `None`;
- `fixed_width_precision_preplanner()` returns `None`;
- no synthetic `4`, `3`, average, or effective bit count is supplied to BE14e.

Those codecs require a future representation-specific quality/storage contract
if they are to participate in a generic Elastic precision policy.

## Composed physical qualification

The consumer tests include one INT4 path where:

1. SLHAv2 emits exact source-bound capacity evidence;
2. ElasticXxx capacity admission returns a candidate plan;
3. the SLHAv2-owned INT4 declaration runs ElasticXxx BE14e precision screening
   with strict DecisionTrace evidence;
4. `RepresentationPrecisionKvBindingV1` proves both paths refer to the same KV
   plan;
5. the existing trusted transaction executes HOT→WARM;
6. physical WARM state is verified and committed;
7. the retained HOT source remains exactly reconstructable for rollback.

This is a correctness/integration qualification. It makes no quality, memory,
throughput, latency, or speedup claim.
