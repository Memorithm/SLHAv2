# KVLab -> SLHAv2 Feedback Contract

Status: active cross-repository research contract, 2026-09-26.

## Purpose

KVLab is the primary upstream bench for KV-state experiments that may improve
SLHAv2. This document prevents a common failure mode: a promising proxy or
logical saving being copied into the runtime before it has demonstrated model
quality and physical benefit.

## Evidence classes

### A. Structural evidence

Examples: exact Hamming parity, candidate-set equality, stale-generation
rejection, bit accounting.

May justify API/test integration only. It does not justify quality, memory or
speed claims.

### B. Mechanistic quality evidence

Required fields where selection/routing is involved:

- dense/reference token or page universe;
- selected survivor IDs;
- top-k overlap/recall for frozen k values;
- boundary pair accuracy or inversion count;
- retained softmax mass;
- omitted softmax mass;
- mass-weighted false-negative metric;
- layer/head/query identity;
- matched-density random and positional controls.

May justify a bounded SLHAv2 development candidate. It does not authorize
protected promotion.

### C. Real-model quality evidence

Required before a SLHAv2 default-quality claim:

- immutable model/tokenizer/dataset revisions or hashes;
- baseline and candidate under identical decoding semantics;
- PPL/NLL/task metric with a preregistered acceptance rule;
- no access to forbidden dense/oracle information in the deployable candidate;
- negative results retained.

### D. Physical systems evidence

Required before speed/memory/bandwidth claims:

- exact hardware/driver/runtime/source revision;
- cache/device residency boundary;
- bytes actually allocated or transferred for the claim;
- TTFT, TPOT/tokens/s and latency distribution;
- warmup/repetition protocol;
- Boolean/front-end and transition overhead included;
- same-regime dense baseline.

Logical bytes avoided are not physical DRAM/HBM traffic.

## Promotion pipeline

```text
KVLab hypothesis
  -> preregister
  -> execute against matched controls
  -> retain positive/negative evidence
  -> freeze producer artifact/schema
  -> SLHAv2 imports as non-default candidate
  -> SLHAv2 destination-owned differential tests
  -> non-protected real-model qualification
  -> freeze candidate
  -> protected gate if applicable
  -> physical qualification
  -> only then consider default promotion
```

## Initial feed-forward variables

The first campaign should expose these variables separately rather than combine
them into one opaque importance score:

- rank position and top-k membership;
- exact attention probability / retained softmax mass as evaluation-only oracle;
- sign/Hamming signature distance;
- key norm and query/key norm interaction;
- token age/position;
- frequency / historical attention mass when available;
- layer and head identity;
- residual energy / current SLHA reconstruction state;
- representation width and transition cost;
- page/block grouping geometry.

The objective is to discover which variables generalize, not to assume all are
useful.

## Mandatory controls

When applicable:

1. full dense attention;
2. paged dense numerical attention;
3. random matched-density;
4. positional/tail matched-density;
5. Boolean/signature-only candidate;
6. norm/structural simple candidate;
7. current SLHA policy;
8. adaptive candidate;
9. non-deployable exact-score/mass oracle only as an upper-bound diagnostic.

## Cross-repository ownership

- KVLab: experiment design, falsification, cache/page metrics, evidence packs.
- SLHAv2: compressed representation, quality budget, destination promotion.
- FLAT-ATTENTION: attention execution and portable GPU timing.
- ElasticXxx: generic adaptation/transactions/rollback.
- NNIS: native NVIDIA execution and physical GPU qualification.
- BooleanLab: Boolean policy/function search.
- Forge/ADA: bounded algorithm/policy search after evaluator freeze.
- SciRust: reusable primitives only after evidence supports promotion.

## First retained upstream evidence — KVLab SKV-1

Producer revision: `90a94e0165a38926835c766a45b4803bc0540317`.

The synthetic SKV-1 panel deliberately holds candidate density constant at
25%. On its controlled dominant-head row, two survivor sets both have top-2
recall = 0.5:

- dropping the strongest key retains softmax mass **0.000638**;
- keeping the strongest key but replacing the weaker top-2 boundary key retains
  mass **0.998532**.

This is not model-quality evidence. It is accepted here only as mechanistic
evidence that set-level top-k recall is insufficient for SLHAv2 routing gates
and that mass-weighted misses must be retained as a separate diagnostic.

## Stop rules

Stop or retain a negative result when:

- a candidate loses to matched controls at equal density/budget;
- quality gains depend on an unavailable dense oracle;
- transition/controller overhead erases the systems advantage;
- results do not reproduce on the frozen validation population;
- physical measurements contradict logical-byte predictions.

A stopped result remains useful evidence and must not be silently recycled under
the same experiment identity.
