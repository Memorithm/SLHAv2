//! Cross-resource example: two local ECAs admitted through one real shared
//! [`ElasticCoordinator`] budget.
//!
//! The controllers retain independent local pressure models, while the
//! coordinator is the single authority for shared byte commitments.

use elastic_core::controller::{
    ControllerConfig, ElasticBackend, ElasticController, Observation,
};
use elastic_core::ElasticResource;
use elastic_runtime::coordinator::ElasticCoordinator;

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
        self.released = self.released.saturating_add(target_bytes);
        Ok(target_bytes)
    }

    fn promote(&mut self, target_bytes: u64) -> Result<u64, Self::Error> {
        self.restored = self.restored.saturating_add(target_bytes);
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

fn set_commitment(
    coordinator: &mut ElasticCoordinator,
    resource: &str,
    current: u64,
    desired: u64,
) -> u64 {
    if desired > current {
        let delta = desired - current;
        if coordinator.commit(resource, delta).is_ok() {
            desired
        } else {
            current
        }
    } else if current > desired {
        let delta = current - desired;
        coordinator
            .release(resource, delta)
            .expect("resource owns its previous commitment");
        desired
    } else {
        current
    }
}

fn main() {
    let mut coordinator = ElasticCoordinator::new();
    let org = coordinator.tree_mut().add_root(0, 100_000, 80_000, 0);
    let tenant = coordinator
        .tree_mut()
        .add_child(org, 5, 60_000, 50_000, 0)
        .unwrap();
    let session = coordinator
        .tree_mut()
        .add_child(tenant, 5, 30_000, 26_000, 0)
        .unwrap();
    let queue_node = coordinator
        .tree_mut()
        .add_child(session, 5, 20_000, 15_000, 4_000)
        .unwrap();
    let workers_node = coordinator
        .tree_mut()
        .add_child(session, 5, 20_000, 15_000, 4_000)
        .unwrap();
    coordinator.register("queue", queue_node, 5).unwrap();
    coordinator.register("workers", workers_node, 5).unwrap();

    // Controllers get the same static topology/limits, while live commitments
    // are admitted through the single coordinator above.
    let topology = coordinator.tree().clone();
    let mut queue = ElasticController::new(
        QueueResource("queue".into()),
        CountingBackend::default(),
        ControllerConfig::standard(),
        topology.clone(),
        Some(queue_node),
    );
    let mut workers = ElasticController::new(
        WorkerResource("workers".into()),
        CountingBackend::default(),
        ControllerConfig::standard(),
        topology,
        Some(workers_node),
    );

    let mut queue_committed = 0u64;
    let mut workers_committed = 0u64;

    for step in 1..=30u64 {
        let queue_desired = (10_000 + step * 500).min(20_000);
        let workers_desired = (8_000 + step * 350).min(20_000);

        // Deterministic admission order for the example. A production policy
        // can schedule requests by priority before calling commit().
        queue_committed = set_commitment(
            &mut coordinator,
            "queue",
            queue_committed,
            queue_desired,
        );
        workers_committed = set_commitment(
            &mut coordinator,
            "workers",
            workers_committed,
            workers_desired,
        );

        let queue_decision = queue
            .step(Observation::new(
                step,
                queue_committed,
                20_000,
                queue_desired as f64,
            ))
            .expect("queue step");
        let worker_decision = workers
            .step(Observation::new(
                step,
                workers_committed,
                20_000,
                workers_desired as f64,
            ))
            .expect("worker step");

        assert!(coordinator.total_committed() <= 30_000);
        println!(
            "step {step:>2} shared={:>5}/30000 queue={:>5}({:<7}) workers={:>5}({:<7})",
            coordinator.total_committed(),
            queue_committed,
            queue_decision.action.name(),
            workers_committed,
            worker_decision.action.name(),
        );
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
    println!(
        "shared coordinator final commitment: {} bytes",
        coordinator.total_committed()
    );
}
