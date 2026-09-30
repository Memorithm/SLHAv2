# SLHAv2 Roadmap — Importance-Preserving Adaptive KV

Status: active research/product roadmap, 2026-09-26.

## North star

SLHAv2 must earn an end-to-end quality/memory/performance win on a real model.
The September 2026 evidence rejects direct compressed-score substitution as the
current production path. The programme therefore pivots from a **single fixed
compressed representation** to a **quality-constrained adaptive KV substrate**.

The optimization order is:

```text
preserve task quality
  -> preserve ranking / top-set / retained softmax mass
  -> reduce numerical KV work and bytes touched
  -> adapt representation / residency / precision
  -> prove physical memory and latency gains
```

No later objective may silently override an earlier one.

## Current evidence boundary

- Qwen2.5-1.5B-Instruct, WikiText-2, matched no-substitution control:
  PPL 11.8831.
- Strict SLHA score replacement: PPL 16.9173 (+42.36%): NO-GO.
- Rank-transplant diagnostic: restoring full baseline ordering recovers 65.9%
  of the quality gap.
- Restoring baseline top-16 ordering captures 98.42% of the full rank-oracle
  recovery.
- TinyStories physical external-K path is K-only; V remains ordinary.
- No end-to-end memory/performance win is currently claimed.

## Architecture target

```text
Decode state
   |
   +--> importance evidence
   |      ranking / top-k / retained softmax mass
   |      Boolean signatures / Hamming
   |      norm / age / frequency / layer / head
   |
   +--> admissible survivor set
   |
   +--> adaptive representation class
          CRITICAL
          IMPORTANT
          ORDINARY
          LOW
          COLD / REPLAY
   |
   +--> numerical attention
   |
   +--> verify quality + memory + latency
   |
   +--> ElasticXxx COMMIT / ROLLBACK
```

These are research states, not a stable public ABI.

## Programme

### RK0 — freeze evidence and repair stale documentation
Status: active.

- keep negative real-model results visible;
- remove stale README statements that perplexity is unmeasured;
- bind all new claims to exact experiment/model/runtime identities;
- fix any accounting/table inconsistency before publication.

Exit: README, paper, FINDINGS and roadmap agree on the same measured facts.

### RK1 — LR1 ranking-preserving projection
Status: active, issue #107.

- execute only the frozen `slha-lr1-pairwise-top16-all-layers-v1` candidate;
- use non-protected training/validation/diagnostic data for Stage A;
- require top-16 overlap/boundary accuracy plus non-protected PPL direction;
- freeze artifact and evaluation command before protected Stage B.

Exit: protected gate accepts or rejects the frozen candidate. No rescue sweep
under the same candidate identity.

### RK2 — KVLab importance map
Status: active; SKV-0 and SKV-1 first slices merged.

First retained upstream evidence: KVLab `90a94e0165a38926835c766a45b4803bc0540317` shows that at equal
25% survivor density and equal top-2 recall 0.5, two controlled selections can
retain 0.000638 versus 0.998532 softmax mass. This does not establish model
quality, but it falsifies top-k recall as a sufficient routing metric.

KVLab becomes the primary research bench for identifying what SLHA must retain.
The initial panel evaluates:

- top-k recall and boundary inversions;
- retained softmax probability mass;
- false-negative cost by mass, not count alone;
- norm/signature/position/age/frequency features;
- layer/head sensitivity;
- matched-density random and positional controls;
- query-agnostic versus query-aware importance.

Exit: at least one policy family has reproducible evidence that improves the
quality/work frontier over matched controls.

### RK3 — Boolean-indexed SLHA
Status: research.

Use KVLab/BooleanLab/FLAT evidence to test a two-stage path:

```text
compact Boolean admission -> exact/SLHA numerical scoring on survivors
```

Dense numerical K/V remains authoritative. No native-Boolean replacement claim
is allowed until BIKV is qualified.

Exit: same-model evidence shows candidate recall/quality within the frozen gate
and measured total routing + survivor attention cost better than dense.

### RK4 — adaptive precision and representation
Status: research.

Candidate dimensions include:

- HOT/WARM/COLD;
- mixed 8/4, INT4, MIX3 and any separately qualified codec;
- residual retention/removal;
- Boolean/index-only metadata;
- elastic control/index widths 64/128/256/512/1024/2048 bits;
- K and V sensitivity handled separately;
- layer/head-specific representation only if calibration evidence justifies it.

ElasticXxx owns the generic transition lifecycle. SLHAv2 owns admissibility,
representation semantics and destination verification.

Exit: adaptive policy beats every required static baseline after observation,
planning, transition, verification and rollback overhead are included.

### RK5 — softmax-mass-aware repair
Status: research.

FLAT MAA evidence shows that top-k count alone is not a sufficient proxy for
softmax quality. Add monotonic repair tiers that can widen a survivor set when a
deployable confidence signal says the current approximation is unsafe.

Rules:

- repair may add survivors, never silently delete already admitted survivors;
- dense exact-score mass is an evaluation oracle, never a deployment trigger;
- compare against matched static-density controls.

Exit: a deployable repair signal improves quality at lower average numerical
work without hidden dense-score access.

### RK6 — full physical K+V integration
Status: blocked by quality.

- keep external-K accounting proof;
- add V only through an explicit versioned representation;
- distinguish cache-owned bytes, process RSS, GPU allocation and traffic;
- preserve baseline fallback.

Exit: real model, same workload: measured K+V memory reduction and quality gate
pass.

### RK7 — native Thor qualification
Status: pending after candidate quality gate.

Use NNIS for:

- device-resident representation;
- CUDA-event timing;
- actual transfer/residency accounting;
- Boolean/index front-end and numerical survivor kernels where justified;
- comparison against the same dense model/runtime path.

Exit: reproducible physical evidence on Thor, not CPU projections.

### RK8 — search and automatic policy synthesis
Status: gated.

After the evaluator is frozen:

- Forge may search representation/policy candidates;
- ADA may search attention algorithms;
- BooleanLab may synthesize compact predicates;
- TDI may study adaptive decision dynamics.

All candidates are proposals. SLHAv2 promotion requires independent
destination-owned verification and protected evaluation where applicable.

## Promotion contract

A mechanism imported from another Memorithm repository must provide:

1. exact source repo/revision and versioned schema;
2. frozen workload/model/tokenizer/runtime identity;
3. authoritative dense/full-fidelity comparator;
4. matched density/budget controls;
5. ranking and retained-softmax-mass diagnostics when selection is involved;
6. model-quality result;
7. exact logical byte accounting;
8. physical timing/traffic evidence for physical claims;
9. fail-closed validation and dense rollback;
10. destination-owned SLHAv2 tests on the exact integration head.

See `docs/KVLAB_FEEDBACK_CONTRACT.md`.

## Publication targets

The paper should present three layers separately:

- **validated mechanism**: tile, Hamming identity, SIMD, Soft-Paging;
- **negative end-to-end evidence**: current direct replacement NO-GO;
- **new falsifiable programme**: ranking/mass-preserving adaptive KV.

A future positive result may be promoted only after it exists. The roadmap is
not evidence.

## CPS-2 destination verification — 2026-09-30

The `scirust::kvlab_cps2` read-only consumer verifies the complete frozen
KVLab #175 development panel against an independent scalar fixture oracle.
The upstream producer is pinned to `265a2a65120b8f6f06c3cbb6604cf821ceda94d0`;
FLAT CPS-1 is pinned to `ad1634fc922f6223dd3a83ac84154a82b1a35562`.
See `docs/CPS2_IMPORT.md` and `docs/CPS2_IMPORT_EVIDENCE.md` for the protocol,
actual execution provenance and limitations. Source/schema drift, missing
controls, forged metrics and unsupported promotion flags are rejected.

This completes a synthetic evidence-consumption slice under RK2/RK3 only.
Real-model quality, physical systems gains and runtime adoption remain gated;
RK1/LR1 and its protected holdout are unchanged.
