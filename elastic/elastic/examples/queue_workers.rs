//! Independent example: ElasticQueue + ElasticWorkers.
//!
//! Proves the generic Elastic language is not secretly specific to KV
//! caches: a work queue whose capacity adapts to pressure, coordinated with
//! a worker pool, using the same ECA controller as SLHAv2's ElasticContext.
//!
//! Run: `cargo run -p elastic --example queue_workers --release`

use elastic::budget::BudgetTree;
use elastic::controller::{
    ActionRequest, ControllerConfig, Decision, ElasticBackend, ElasticController, Observation,
};
use elastic::ElasticResource;

/// A bounded work queue resource.
#[derive(Debug, Clone)]
pub struct WorkQueue {
    id: String,
}

impl WorkQueue {
    /// Create a queue.
    pub fn new(id: &str, _capacity: u64) -> Self {
        Self { id: id.to_string() }
    }
}

impl ElasticResource for WorkQueue {
    fn resource_id(&self) -> &str {
        &self.id
    }
}

/// The queue backend: enqueue/dequeue + demote (drop queued work) + promote
/// (restore capacity).
#[derive(Debug, Default)]
pub struct QueueBackend {
    /// Dropped queued items (demote).
    pub dropped: u64,
    /// Re-admitted items (promote).
    pub admitted: u64,
}

impl ElasticBackend for QueueBackend {
    type Error = &'static str;

    fn demote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.dropped += target_bytes;
        Ok(target_bytes)
    }

    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.admitted += target_bytes;
        Ok(target_bytes)
    }

    fn offload(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.demote(target_bytes)
    }

    fn restore(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.promote(target_bytes)
    }

    fn prefetch(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn rebalance(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn verify(&mut self, _expected_used: u64) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

fn main() {
    let budget = BudgetTree::new();
    let mut controller = ElasticController::new(
        WorkQueue::new("ingest", 1000),
        QueueBackend::default(),
        ControllerConfig::standard(),
        budget,
        None,
    );

    // Simulated workload: the queue grows until it approaches capacity
    // (pressure High), then drains.
    println!("step  action  pressure");
    for step in 1..=40u64 {
        let used = if step <= 15 {
            400 + step * 40 // growing: 440 .. 1000
        } else {
            1000u64.saturating_sub((step - 15) * 60) // draining
        };
        let used = used.min(1000);
        let d: Decision = controller
            .step(Observation::new(step, used, 1000, 10.0))
            .expect("controller step");
        let action = d.action.name();
        let p = d.trace.measured_pressure;
        println!("{step:>4}  {action:<8} {p:.3}");
        if d.action != ActionRequest::None {
            println!("        -> {}", d.trace.explain());
        }
    }

    println!(
        "\nqueue dropped {} items, admitted {} items",
        controller.backend().dropped,
        controller.backend().admitted
    );
    println!("elastic ECA demo complete (deterministic).");
}
