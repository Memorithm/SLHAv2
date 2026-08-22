//! Macro semantics integration tests: generated syntax must perform real work
//! or be explicitly descriptive metadata.

use elastic::elastic_budget;
use elastic::elastic_policy;
use elastic::elastic_state;
use elastic::elastic_target;

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
    assert!(tier.try_move(ContextTier::Cold).is_ok());
}

#[test]
fn state_machine_rejects_undeclared_edges() {
    let mut tier = ContextTier::Warm;
    assert!(tier.try_move(ContextTier::Evicted).is_err());
    assert_eq!(tier, ContextTier::Warm);
    let mut hot = ContextTier::Hot;
    assert!(hot.try_move(ContextTier::Cold).is_err());
    assert_eq!(hot, ContextTier::Hot);
}

#[test]
fn pinned_cannot_be_evicted() {
    let machine = ContextTier::machine();
    let pinned = machine.find("Pinned").unwrap();
    let evicted = machine.find("Evicted").unwrap();
    assert!(machine.transition(pinned, evicted).is_err());
}

#[test]
fn budget_macro_evaluates_declared_limits() {
    let budget = ElasticBudget {
        vram: 0.5,
        ram: 0.3,
    };
    assert!(budget.all_satisfied());
    assert_eq!(budget.len(), 2);
    assert!(!budget.is_empty());

    let too_large = ElasticBudget {
        vram: 0.9,
        ram: 0.3,
    };
    assert!(!too_large.all_satisfied());

    let non_finite = ElasticBudget {
        vram: f64::NAN,
        ram: 0.3,
    };
    assert!(!non_finite.all_satisfied());
}

#[test]
fn policy_macro_exposes_validated_metadata() {
    let policy = ContextPolicy::build();
    assert_eq!(policy.correctness, "required");
    assert_eq!(policy.pinned, "preserved");
    assert_eq!(
        policy.objectives,
        ["maximize_retention", "minimize_latency"]
    );
    assert_eq!(policy.hysteresis_high, 0.85);
    assert_eq!(policy.hysteresis_low, 0.70);
    assert!(policy.predictive);
    assert!(policy.transactional);
    assert_eq!(policy.hard_len(), 2);
    assert_eq!(policy.objective_len(), 2);
}

#[test]
fn target_macro_evaluates_objective_and_constraints() {
    let logical_context = 8192_u64;
    let vram_pressure = 0.72_f64;
    let latency_ms = 18.0_f64;
    let target_latency_ms = 25.0_f64;

    let target = elastic_target! {
        maximize logical_context,
        subject_to {
            vram_pressure < 0.85,
            latency_ms <= target_latency_ms,
        }
    };

    assert_eq!(target.objective, 8192.0);
    assert_eq!(target.constraints, [true, true]);
    assert!(target.feasible());
    assert_eq!(target.violations(), 0);
}

#[test]
fn target_macro_reports_failed_constraints() {
    let score = 42.0_f64;
    let pressure = 0.90_f64;
    let quality_ok = true;

    let target = elastic_target! {
        maximize score,
        subject_to {
            pressure < 0.85,
            quality_ok,
        }
    };

    assert_eq!(target.objective, 42.0);
    assert_eq!(target.constraints, [false, true]);
    assert!(!target.feasible());
    assert_eq!(target.violations(), 1);
}
