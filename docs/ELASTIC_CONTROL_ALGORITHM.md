# Elastic Control Algorithm (ECA)

The Elastic architecture is driven by one generic control algorithm. It is
implemented in `elastic-core/src/controller.rs` and shared by every
`ElasticXxx` resource.

## Formal cycle

```text
OBSERVE
   ↓
MODEL
   ↓
PREDICT
   ↓
OPTIMIZE
   ↓
ACT
   ↓
VERIFY
   ↓
LEARN / UPDATE HISTORY
   ↓
REPEAT
```

## Conceptual state

| Symbol | Meaning |
|---|---|
| `S(t)` | current resource/system state |
| `D(t)` | current demand |
| `D̂(t+Δ)` | forecast demand |
| `R(t)` | available physical resources |
| `P(t)` | current/forecast pressure |
| `Q(t)` | correctness/quality state |
| `C(a)` | transition cost of action `a` |

## Hard constraints first

A candidate action is only considered after HARD constraints pass. Hard
constraints have priority over utility optimization:

- memory safety;
- correctness invariants;
- model positional limits;
- PINNED resource guarantees;
- tenant boundaries;
- real physical capacity;
- format/ABI rules.

## Utility

Conceptual utility of an action:

```text
Utility(a) =
      expected_resource_benefit
    + expected_performance_benefit
    + expected_quality_benefit
    + future_headroom_benefit
    - transition_cost
    - restore_cost
    - quality_risk
    - oscillation_penalty
    - future_penalty
```

The v1 controller does NOT encode this as one floating-point weighted sum.
It uses lexicographic, hard-constrained decisions (critical pressure
overrides everything; the hysteresis gate arbitrates; the forecast triggers
pre-emptive demotion), which is safer than a weighted sum.

## Pressure

Pressure is normalized to `[0, 1]` and exposed at four levels:

```text
Low  ≤ low_watermark
Normal
High  ≥ high_watermark
Critical = used > capacity  (hard condition)
```

**Critical overrides weighted averages**: imminent OOM is Critical even if
latency and throughput are excellent.

## Forecasting (v1, deterministic)

- EWMA level + EWMA trend (`elastic-core/src/forecast.rs`);
- bounded derivative (`max_trend`) so a single spike cannot produce an absurd
  forecast;
- `steps_to_exhaustion(forecast, capacity)` drives pre-emptive action.

No machine learning is required for the baseline controller. Learned
policies belong in CCOS Research Lab or an explicitly experimental feature.

## Anti-thrash

The controller never alternates compress/decompress around one noisy
threshold:

- HIGH watermark (default 0.85) enters the demote state;
- LOW watermark (default 0.70) exits it;
- cooldown steps (default 3) block opposite actions;
- min-interval steps (default 2) block rapid re-entry;
- the pressure signal is EWMA-smoothed before any comparison.

## Transactional transitions

Any adaptation that can destroy or invalidate state uses:

```text
PREPARE new representation
    ↓
VALIDATE new representation
    ↓
COMMIT pointer/state switch
    ↓
RELEASE old representation
```

On failure: ROLLBACK. The old representation remains usable until commit.
`elastic-core/src/transaction.rs` implements the executor; a failed rollback
is reported as `RollbackFailed` (a hard error, never silently swallowed).

## Determinism

Same observations + same budgets + same history + same policy + same
config ⇒ same decisions. Wall-clock time is never read by the core; the
controller uses logical steps. See `docs/ELASTIC_SAFETY_INVARIANTS.md`.
