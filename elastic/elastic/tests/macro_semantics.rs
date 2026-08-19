//! Macro semantics integration tests: the generated code must lower into the
//! real elastic-core runtime (validated tier tables, checked transitions).

use elastic::elastic_budget;
use elastic::elastic_policy;
use elastic::elastic_state;
use elastic::prelude::*;

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
        Pinned => !Evicted,
    }
}

elastic_budget! {
    vram <= 0.80,
    ram <= 0.70,
}

elastic_policy! {
    ContextPolicy {
        hard { correctness: "required", pinned: "preserved" }
        objectives { "maximize_retention", "minimize_latency" }
        hysteresis { high: 0.85, low: 0.70 }
        predictive: true
        transactional: true
    }
}

#[test]
fn state_machine_allows_declared_edges() {
    let mut tier = ContextTier::Hot;
    assert!(tier.try_move(ContextTier::Warm).is_ok());
    assert_eq!(tier, ContextTier::Warm);
    assert!(tier.try_move(ContextTier::Hot).is_ok());
    assert!(tier.try_move(ContextTier::Warm).is_ok());
    assert!(tier.try_move(ContextTier::Cold).is_ok());
    assert!(tier.try_move(ContextTier::Evicted).is_ok());
    // Evicted -> Cold is declared; Cold -> Hot is not.
    assert!(tier.try_move(ContextTier::Cold).is_ok());
}

#[test]
fn state_machine_rejects_undeclared_edges() {
    let mut tier = ContextTier::Warm;
    assert!(tier.try_move(ContextTier::Evicted).is_err());
    assert_eq!(tier, ContextTier::Warm); // unchanged on failure
    let mut hot = ContextTier::Hot;
    assert!(hot.try_move(ContextTier::Cold).is_err());
    assert_eq!(hot, ContextTier::Hot);
}

#[test]
fn pinned_cannot_be_evicted() {
    // `Pinned => !Evicted` is a compile-time validated negative edge; at
    // runtime the machine has no Pinned->Evicted transition.
    let machine = ContextTier::machine();
    let pinned = machine.find("Pinned").unwrap();
    let evicted = machine.find("Evicted").unwrap();
    assert!(machine.transition(pinned, evicted).is_err());
}

#[test]
fn budget_macro_generates_real_struct() {
    let b = ElasticBudget {
        vram: 0.5,
        ram: 0.3,
    };
    assert!(b.all_satisfied());
    let bad = ElasticBudget {
        vram: 0.9,
        ram: 0.3,
    };
    assert!(!bad.all_satisfied());
    assert_eq!(
        ElasticBudget {
            vram: 0.0,
            ram: 0.0
        }
        .len(),
        2
    );
}

#[test]
fn policy_macro_generates_real_struct() {
    let p = ContextPolicy::build();
    assert_eq!(p.correctness, "required");
    assert_eq!(p.objectives, ["maximize_retention", "minimize_latency"]);
    assert_eq!(p.hysteresis_high, 0.85);
    assert_eq!(p.hysteresis_low, 0.70);
    assert!(p.predictive);
    assert!(p.transactional);
    assert_eq!(p.hard_len(), 2);
    assert_eq!(p.objective_len(), 2);
}
