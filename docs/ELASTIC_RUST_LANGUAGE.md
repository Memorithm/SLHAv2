# Elastic Rust Language

An embedded adaptive-resource programming language for Rust, implemented
with normal stable Rust mechanisms: traits, generics, strong types,
declarative macros, procedural macros, and a deterministic runtime.

**Do not ask the programmer to guess one permanent resource value when the
system can safely observe, optimize, adapt and verify that value
continuously.**

## Crates

| Crate | Role | Dependencies |
|---|---|---|
| `elastic-core` | generic traits/types, ECA, no unsafe, `no_std + alloc` | none |
| `elastic-runtime` | observers, telemetry, journals, coordination | elastic-core |
| `elastic-macros` | procedural macros with real semantics | proc-macro2/quote/syn |
| `elastic-testkit` | deterministic simulator, fault injection | elastic-core, elastic-runtime |
| `elastic` | public facade + prelude | the four above |

Dependency direction: `elastic-core ← elastic-runtime ← elastic ← SLHAv2 ←
integrations`. The language never depends on SLHAv2.

## Core types

```rust
ElasticResource      // a named resource (id + Debug)
ElasticCapabilities  // can_release / can_restore / can_predict / is_transactional
ElasticConstraints   // hard-constraint validation (priority over utility)
ElasticValue<T>      // Fixed / Auto / Adaptive{min,max} / Pinned
ElasticBudget        // hierarchical: reservations, hard limits, borrowing
ElasticPressure      // normalized [0,1] + Low/Normal/High/Critical
ElasticForecast      // deterministic EWMA + bounded trend
ElasticPolicy        // hard constraints, objectives, hysteresis, flags
ElasticAction        // Demote/Promote/Offload/Restore/Prefetch/Rebalance
ElasticDecision      // action + DecisionTrace (explainable)
ElasticTransition    // prepare/validate/commit/release + rollback
ElasticOutcome       // Committed / PrepareFailed / ValidationFailed / …
ElasticTelemetry     // bytes, pressure, counters, latency
ElasticController    // the generic ECA
ElasticCoordinator   // shared hierarchical budgets across resources
```

### `ElasticValue`

```rust
enum ElasticValue<T> {
    Fixed(T),                    // operator explicitly fixes it
    Auto,                        // ECA owns it CONTINUOUSLY
    Adaptive { min: T, max: T }, // ECA chooses inside the legal range
    Pinned(T),                   // protected from adaptation
}
```

`Auto` NEVER means "choose a default once". It means the runtime
continuously owns this decision.

## Embedded syntax (procedural macros)

### `elastic_state!`

```rust
elastic_state! {
    ContextTier {
        Pinned, Hot, Warm, Cold, Evicted,
    }
    transitions {
        Hot => Warm,
        Warm => Hot,
        Warm => Cold,
        Cold => Warm,
        Cold => Evicted,
        Evicted => Cold,
        Pinned => !Evicted,   // compile-time-validated negative edge
    }
}
```

Generates the enum, the validated `TierMachine`, `TierLike` impls and a
checked `try_move` returning `Result`. Undeclared edges fail at runtime with
the state unchanged; a negative edge (`Pinned => !Evicted`) is checked at
compile time and is also absent from the runtime table.

### `elastic_budget!`

```rust
elastic_budget! {
    vram <= 0.80,
    ram  <= 0.70,
}
```

Generates a typed budget struct with an `all_satisfied()` check.

### `elastic_policy!`

```rust
elastic_policy! {
    ContextPolicy {
        hard { correctness: "required", pinned: "preserved" }
        objectives { "maximize_retention", "minimize_latency" }
        hysteresis { high: 0.85, low: 0.70 }
        predictive: true
        transactional: true
    }
}
```

Generates a policy struct whose fields drive the controller configuration.

### `elastic_target!`

```rust
elastic_target! {
    maximize logical_context,
    subject_to {
        vram_pressure < 0.85,
        latency <= target_latency,
    }
}
```

### `elastic_transition!` (runtime seam)

Transactional transitions are written directly against
`elastic_core::transaction::{Transaction, run_transaction}` — prepare /
validate / commit / rollback closures with the executor guaranteeing that
the old representation survives every failure before commit.

These macros are NOT decorative aliases: they lower into the actual
`elastic-core` runtime abstractions, and the test suite (`elastic/tests/
macro_semantics.rs`) proves the generated code behaves like the real types.

## Examples

- `elastic/examples/queue_workers.rs` — ElasticQueue (a work queue) adapts
  to pressure: demotes queued work under High pressure, promotes after
  recovery. Proves the language is not KV-cache-specific.
- `elastic-testkit/examples/cross_resource.rs` — ElasticQueue +
  ElasticWorkers share one coordinator budget and never fight.

## Honest limits of the v1 language

- Compile-time rejection is provided where the transition graph is known
  statically; dynamic state always returns a checked `Result` against a
  validated table. We do not claim stronger guarantees than Rust provides.
- The controller is deterministic and lexicographic (hard constraints,
  hysteresis, forecast) rather than a learned utility function. Learned
  policies are a Research Lab concern.
