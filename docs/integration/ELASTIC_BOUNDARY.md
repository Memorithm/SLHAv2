# SLHAv2 / Elastic compatibility and ownership boundary

Version: `slhav2-elastic-boundary/v1`.

This document qualifies the two intentionally distinct Elastic dependency spaces
consumed by `slhav2-vram`. Their coexistence is not a Cargo package collision:
the local package is `elastic-core`; the optional external facade is
`memorithm-elastic`, with its own source identity and package family.

The machine-readable source of truth is
[`slhav2-elastic-boundary-v1.json`](slhav2-elastic-boundary-v1.json). CI checks
that file against the manifests, lockfile and resolved Cargo graphs.

## Current boundary

| Boundary | Exact identity | Authority | Runtime status |
| --- | --- | --- | --- |
| Local incubator | `../elastic/elastic-core` 0.1.0 | SLHAv2 preserves the legacy cache/control semantics it currently consumes | Active in the default `slhav2-vram` graph |
| External facade | `memorithm-elastic` 0.1.0 from ElasticXxx `4900a4ae…` | ElasticXxx owns the generic adaptation lifecycle | Optional behind `elasticxxx`; not default control authority |

The nested `elastic/` workspace remains excluded from the root workspace and
has its own lockfile and validation surface. The root lockfile must identify
the local package without an external source and the optional external package
at the exact recorded revision.

## Extraction and retirement gate

The local incubator must not be deleted or silently replaced merely because the
external graph compiles. Retirement requires a scoped, versioned migration:

1. Inventory every local state, budget, journal, transaction and rollback
   semantic actually consumed by SLHAv2.
2. Map that inventory to an explicit ElasticXxx contract without transferring
   SLHA tile, codec, residency, eviction or quality authority.
3. Run destination-owned conformance fixtures on identical traces against both
   implementations. Compare admissibility, state transitions, byte accounting,
   commit outcomes and rollback outcomes.
4. Preserve the local and dense/full-fidelity fallbacks until the candidate
   passes exact-head CI and destination qualification.
5. For product claims, measure the same real model, tokenizer, workload and
   runtime for quality, physical memory and end-to-end throughput.
6. Update the matrix, lockfile, evidence and roadmap in the same reviewed
   migration change.

A failed or incomplete parity gate leaves the local path in place. A compile
check, dependency tree, synthetic trace or software GPU backend is not evidence
of real-model parity or physical performance.

## Ownership

SLHAv2 continues to own compressed-KV formats, codecs, cache-specific
HOT/WARM/COLD semantics, eviction, quality gates and destination rollback.
ElasticXxx owns reusable generic observe/plan/validate/act/verify/commit
orchestration. Neither origin gains authority over the other merely by appearing
in the same resolved graph.
