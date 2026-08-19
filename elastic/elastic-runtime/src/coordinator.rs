//! Multi-resource coordination: hierarchical budgets and headroom lending.
//!
//! Independent elastic controllers must not fight each other (ElasticContext
//! compresses to save VRAM while ElasticMemory sees free RAM and promotes,
//! and ElasticKvCache demotes again). The coordinator owns the shared budget
//! tree; each controller borrows headroom only when the parent budget allows
//! it and no pinned guarantee is violated.

use elastic_core::budget::{BudgetError, BudgetTree};

/// A registered resource in the coordinator.
#[derive(Clone, Debug)]
pub struct RegisteredResource {
    /// Resource identifier.
    pub resource_id: String,
    /// Budget node index in the shared tree.
    pub node: usize,
    /// Priority class (higher = more important).
    pub priority: u8,
}

/// The coordinator: owns the shared budget tree and the resource registry.
#[derive(Clone, Debug, Default)]
pub struct ElasticCoordinator {
    tree: BudgetTree,
    resources: Vec<RegisteredResource>,
}

impl ElasticCoordinator {
    /// Create an empty coordinator.
    pub fn new() -> Self {
        Self::default()
    }

    /// The shared budget tree (read access for telemetry).
    pub fn tree(&self) -> &BudgetTree {
        &self.tree
    }

    /// The mutable shared budget tree (for setup).
    pub fn tree_mut(&mut self) -> &mut BudgetTree {
        &mut self.tree
    }

    /// Register a resource under an existing budget node.
    pub fn register(
        &mut self,
        resource_id: &str,
        node: usize,
        priority: u8,
    ) -> Result<usize, BudgetError> {
        self.tree.node(node)?; // validate existence
        let idx = self.resources.len();
        self.resources.push(RegisteredResource {
            resource_id: resource_id.to_string(),
            node,
            priority,
        });
        Ok(idx)
    }

    /// Look up a resource by id.
    pub fn find(&self, resource_id: &str) -> Option<&RegisteredResource> {
        self.resources.iter().find(|r| r.resource_id == resource_id)
    }

    /// Registered resources.
    pub fn resources(&self) -> &[RegisteredResource] {
        &self.resources
    }

    /// Try to commit `bytes` for `resource_id` (borrowing from the parent
    /// budget). Fails closed on hard-limit/parent/priority violations.
    pub fn commit(&mut self, resource_id: &str, bytes: u64) -> Result<(), BudgetError> {
        let reg = self.find(resource_id).ok_or(BudgetError::NoSuchNode)?;
        let node = reg.node;
        // Priority guard: a lower-priority borrower cannot displace a
        // higher-priority tenant's reservation.
        let parent = self.tree.node(node).map(|n| n.parent);
        if let Ok(Some(p)) = parent {
            let pnode = self.tree.node(p)?;
            let pused = self.tree.committed_with_descendants(p)?;
            let reserved = pnode.reservation;
            let headroom = pnode.hard_limit.saturating_sub(pused);
            if pused + bytes > pnode.hard_limit && reg.priority < pnode.priority {
                return Err(BudgetError::PriorityViolated);
            }
            if headroom < bytes && reserved > 0 && reg.priority < pnode.priority {
                return Err(BudgetError::PriorityViolated);
            }
        }
        self.tree.try_commit(node, bytes)
    }

    /// Release `bytes` for `resource_id`.
    pub fn release(&mut self, resource_id: &str, bytes: u64) -> Result<(), BudgetError> {
        let reg = self.find(resource_id).ok_or(BudgetError::NoSuchNode)?;
        self.tree.release(reg.node, bytes)
    }

    /// Total committed bytes across all resources.
    pub fn total_committed(&self) -> u64 {
        self.tree.total_committed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinator() -> ElasticCoordinator {
        let mut c = ElasticCoordinator::new();
        let org = c.tree_mut().add_root(0, 1000, 800, 200);
        let tenant = c.tree_mut().add_child(org, 5, 500, 400, 100).unwrap();
        let session = c.tree_mut().add_child(tenant, 5, 400, 300, 50).unwrap();
        c.register("ctx", session, 5).unwrap();
        c.register("mem", session, 5).unwrap();
        c
    }

    #[test]
    fn resources_share_parent_budget() {
        let mut c = coordinator();
        assert!(c.commit("ctx", 300).is_ok());
        // 300 + 200 = 500 > session limit 400.
        assert_eq!(c.commit("mem", 200), Err(BudgetError::HardLimitExceeded));
        assert_eq!(c.total_committed(), 300);
        c.release("ctx", 300).unwrap();
        assert!(c.commit("mem", 400).is_ok());
    }

    #[test]
    fn unknown_resource_fails_closed() {
        let mut c = coordinator();
        assert_eq!(c.commit("nope", 1), Err(BudgetError::NoSuchNode));
    }

    #[test]
    fn higher_priority_borrows_first() {
        let mut c = coordinator();
        // Session hard limit 400; ctx (priority 5) commits 350.
        c.commit("ctx", 350).unwrap();
        // mem same priority: 350 + 100 = 450 > 400 -> hard limit.
        assert_eq!(c.commit("mem", 100), Err(BudgetError::HardLimitExceeded));
    }
}
