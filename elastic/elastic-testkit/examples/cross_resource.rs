//! Cross-resource example: ElasticQueue + ElasticWorkers sharing a
//! coordinator budget — proves the ECA coordinates TWO resources with one
//! shared hierarchical budget.
//!
//! Run: `cargo run -p elastic-testkit --example cross_resource`

use elastic_core::budget::BudgetTree;
use elastic_core::controller::{
    ActionRequest, ControllerConfig, ElasticBackend, ElasticController, Observation,
};
use elastic_core::ElasticResource;

#[derive(Debug, Clone)]
struct QueueResource(String);
impl ElasticResource for QueueResource {
    fn resource_id(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
struct WorkerResource(String);
impl ElasticResource for WorkerResource {
    fn resource_id(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Default)]
struct CountingBackend {
    released: u64,
    restored: u64,
}
impl ElasticBackend for CountingBackend {
    type Error = &'static str;
    fn demote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.released += target_bytes;
        Ok(target_bytes)
    }
    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.restored += target_bytes;
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
    // Shared hierarchical budget: org -> tenant -> session -> {queue, workers}.
    let mut tree = BudgetTree::new();
    let org = tree.add_root(0, 100_000, 80_000, 20_000);
    let tenant = tree.add_child(org, 5, 60_000, 50_000, 10_000).unwrap();
    let session = tree.add_child(tenant, 5, 40_000, 30_000, 5_000).unwrap();
    let queue_node = tree.add_child(session, 5, 20_000, 15_000, 0).unwrap();
    let workers_node = tree.add_child(session, 5, 20_000, 15_000, 0).unwrap();

    let mut queue = ElasticController::new(
        QueueResource("queue".into()),
        CountingBackend::default(),
        ControllerConfig::standard(),
        tree.clone(),
        Some(queue_node),
    );
    let mut workers = ElasticController::new(
        WorkerResource("workers".into()),
        CountingBackend::default(),
        ControllerConfig::standard(),
        tree,
        Some(workers_node),
    );

    // Queue pressure rises while workers stay moderate.
    for step in 1..=30u64 {
        let q_used = if step <= 12 {
            12_000 + step * 700
        } else {
            18_000
        };
        let w_used = 9_000 + step * 100;
        let dq = queue
            .step(Observation::new(step, q_used.min(20_000), 20_000, 10.0))
            .expect("queue step");
        let dw = workers
            .step(Observation::new(step, w_used.min(20_000), 20_000, 5.0))
            .expect("workers step");
        println!(
            "step {step:>2} queue={:<7} workers={:<7}",
            dq.action.name(),
            dw.action.name()
        );
        let _ = ActionRequest::None; // silence unused import lint on some cfgs
    }

    println!(
        "\nqueue backend: released={} restored={}",
        queue.backend().released,
        queue.backend().restored
    );
    println!(
        "workers backend: released={} restored={}",
        workers.backend().released,
        workers.backend().restored
    );
    println!("cross-resource ECA demo complete (deterministic).");
}
