//! elastic-testkit — deterministic simulator, scripted pressure, fault
//! injection, state-machine helpers.
//!
//! The testkit drives [`ElasticController`] with scripted observation
//! sequences and scripted backend failures, so invariants can be proven
//! under pressure spikes, sustained pressure, allocation failure, and
//! repeated grow/shrink cycles — all deterministically.

use elastic_core::controller::{
    ActionRequest, Decision, ElasticBackend, ElasticController, Observation,
};
use elastic_core::ElasticResource;

/// A scripted backend that records executed actions and can be programmed to
/// fail specific operations (fault injection).
#[derive(Debug, Default)]
pub struct ScriptedBackend {
    /// Executed actions in order.
    pub actions: Vec<&'static str>,
    /// Bytes actually released per demote/offload.
    pub released: u64,
    /// Bytes actually restored per promote/restore.
    pub restored: u64,
    /// Fail the next `demote` call.
    pub fail_demote: bool,
    /// Fail the next `promote` call.
    pub fail_promote: bool,
    /// Fail verification.
    pub fail_verify: bool,
}

impl ScriptedBackend {
    /// Create an empty scripted backend.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ElasticBackend for ScriptedBackend {
    type Error = &'static str;

    fn demote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        if self.fail_demote {
            self.fail_demote = false;
            return Err("scripted demote failure");
        }
        self.actions.push("demote");
        self.released += target_bytes;
        Ok(target_bytes)
    }

    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        if self.fail_promote {
            self.fail_promote = false;
            return Err("scripted promote failure");
        }
        self.actions.push("promote");
        self.restored += target_bytes;
        Ok(target_bytes)
    }

    fn offload(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.actions.push("offload");
        self.released += target_bytes;
        Ok(target_bytes)
    }

    fn restore(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.actions.push("restore");
        self.restored += target_bytes;
        Ok(target_bytes)
    }

    fn prefetch(&mut self) -> Result<(), Self::Error> {
        self.actions.push("prefetch");
        Ok(())
    }

    fn rebalance(&mut self) -> Result<(), Self::Error> {
        self.actions.push("rebalance");
        Ok(())
    }

    fn verify(&mut self, _expected_used: u64) -> Result<bool, Self::Error> {
        if self.fail_verify {
            self.fail_verify = false;
            return Err("scripted verify failure");
        }
        Ok(true)
    }
}

/// A minimal resource for the testkit.
#[derive(Debug, Clone)]
pub struct ScriptedResource {
    id: String,
}

impl ScriptedResource {
    /// Create a resource with an id.
    pub fn new(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl ElasticResource for ScriptedResource {
    fn resource_id(&self) -> &str {
        &self.id
    }
}

/// A deterministic simulator: feeds a scripted observation sequence to a
/// controller and collects the decisions.
pub struct ElasticSimulator<R, B> {
    controller: ElasticController<R, B>,
    /// Collected decisions, in order.
    pub decisions: Vec<Decision>,
}

impl<R: ElasticResource, B: ElasticBackend> ElasticSimulator<R, B> {
    /// Create a simulator.
    pub fn new(controller: ElasticController<R, B>) -> Self {
        Self {
            controller,
            decisions: Vec::new(),
        }
    }

    /// Run a scripted observation sequence.
    pub fn run(&mut self, script: &[Observation]) -> Result<(), B::Error> {
        for obs in script {
            let d = self.controller.step(obs.clone())?;
            self.decisions.push(d);
        }
        Ok(())
    }

    /// The chosen actions in order (for assertions).
    pub fn actions(&self) -> Vec<&'static str> {
        self.decisions.iter().map(|d| d.action.name()).collect()
    }

    /// Number of times the given action was chosen.
    pub fn count(&self, action: ActionRequest) -> usize {
        self.decisions.iter().filter(|d| d.action == action).count()
    }

    /// Access to the controller (telemetry).
    pub fn controller(&self) -> &ElasticController<R, B> {
        &self.controller
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elastic_core::budget::BudgetTree;
    use elastic_core::controller::ControllerConfig;

    fn ctl() -> ElasticController<ScriptedResource, ScriptedBackend> {
        let budget = BudgetTree::new();
        ElasticController::new(
            ScriptedResource::new("test"),
            ScriptedBackend::new(),
            ControllerConfig::standard(),
            budget,
            None,
        )
    }

    #[test]
    fn sustained_pressure_demotes_once_then_stays_active() {
        let mut sim = ElasticSimulator::new(ctl());
        let script: Vec<Observation> = (1..=10)
            .map(|s| Observation::new(s, 900, 1000, 10.0))
            .collect();
        sim.run(&script).unwrap();
        // The gate activates once (high pressure) and stays active; no
        // oscillating demote/promote alternation.
        assert_eq!(sim.count(ActionRequest::Demote), 1);
        assert_eq!(sim.count(ActionRequest::Promote), 0);
    }

    #[test]
    fn pressure_spike_then_recovery_promotes_after_cooldown() {
        let mut sim = ElasticSimulator::new(ctl());
        let mut script = vec![];
        for s in 1..=5 {
            script.push(Observation::new(s, 900, 1000, 10.0));
        }
        for s in 6..=30 {
            script.push(Observation::new(s, 100, 1000, 1.0));
        }
        sim.run(&script).unwrap();
        assert_eq!(sim.count(ActionRequest::Demote), 1);
        // Promotion happens only after the low watermark is restored, and
        // only a few times at most across the whole recovery episode
        // (anti-thrash: the controller must not promote every step).
        let promotions = sim.count(ActionRequest::Promote);
        assert!((1..=5).contains(&promotions), "promotions={promotions}");
    }
}
