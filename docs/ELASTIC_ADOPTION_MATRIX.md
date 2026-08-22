# Elastic Adoption Matrix

Cross-project assessment: which subsystems are genuine Elastic candidates.
`ElasticXxx` is NOT a naming convention — it is a generic constrained
adaptive-resource control system. A candidate qualifies only when its
operating point varies, runtime observations inform better decisions,
adaptation has measurable benefit, and adaptation cost is lower than the
benefit.

Legend: **USE ELASTIC NOW** · **CANDIDATE LATER** · **NOT APPROPRIATE** ·
**ALREADY EFFECTIVELY ELASTIC** · **NEEDS RESEARCH**

## SLHAv2

| Subsystem | Verdict | Reason |
|---|---|---|
| Context residency (`ElasticContext`, new) | USE ELASTIC NOW | The core doctrine: context is a runtime resource. Implemented on the generic ECA in this mission. |
| KV cache residency (`ElasticKvCache`) | USE ELASTIC NOW | Existing soft-paging is logical accounting; must become physical residency on the generic engine (P0-E1). |
| Memory / VRAM arena | CANDIDATE LATER | The arena allocator itself is a fixed-capacity pool; elasticity applies to how the pool is sized and shared, not to the allocator math. |
| Tile codec constants (128 B, D_C=128) | NOT APPROPRIATE | `AlgorithmInvariant`/`FormatInvariant` — fixed by the math and the persisted format. |
| Layer/head/token caps in llama shim | NOT APPROPRIATE (as elastic) | They are `LegacyArbitraryLimit`s that must be REMOVED (derived from runtime metadata), not made adaptive. |
| Score replacement quality | NEEDS RESEARCH | Quality-aware elasticity requires a validated quality signal; the ranking research is not yet deployable (see `results/`). |

## CCOS Core

| Subsystem | Verdict | Reason |
|---|---|---|
| Working-set residency / context budgeting | USE ELASTIC NOW (via ports) | Core owns causal truth; Elastic provides observations + storage capability through the memory-provider boundary, never authoritative causal state. |
| Token budget adaptation | CANDIDATE LATER | Requires the provider boundary and deterministic replay to be exercised first. |
| EventLog / paging I/O | ALREADY EFFECTIVELY ELASTIC | Core's soft-paging + EventLog is a working elasticity loop; align it with the generic ECA rather than rewriting. |

## CCOS Enterprise

| Subsystem | Verdict | Reason |
|---|---|---|
| Hierarchical budgets (org → tenant → session) | USE ELASTIC NOW (as a model) | The `BudgetTree` maps 1:1 onto tenant quotas; Enterprise governs Elastic, never duplicates its algorithm. |
| Quota enforcement | NOT APPROPRIATE (as elastic) | Hard quota overrides local optimization; it is a hard constraint, not an adaptive target. |

## CCOS Research Lab

| Subsystem | Verdict | Reason |
|---|---|---|
| Learned predictors / RL policies | USE ELASTIC NOW (as the lab sandbox) | Experimental policies belong outside the certifiable boundary; the ECA has a clean seam for them. |

## FLAT-ATTENTION

| Subsystem | Verdict | Reason |
|---|---|---|
| Device capability fingerprint | ALREADY EFFECTIVELY ELASTIC | Runtime capability fingerprinting for autotuning exists; adapt it into `ElasticCapabilities` via an adapter (name is provenance, never policy). |

## Not elastic, ever

- mathematical constants (`D_C`, tile geometry);
- file-format invariants (tile offsets, `slhw` layout);
- ABI invariants (`SciRustSlhaTile`, `slha_abi_version`);
- protocol constants (MCP JSON-RPC, C status codes);
- true model limits (RoPE max context);
- true hardware limits (VRAM size).
