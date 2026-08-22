//! Multi-resource coordination over one shared hierarchical budget tree.
//!
//! Controllers may make independent local decisions, but all shared-resource
//! admission/release must pass through this coordinator. Per-resource
//! commitments are tracked separately even when several resources intentionally
//! share the same budget node, so one resource can never release another's
//! bytes.

use core::fmt;

use elastic_core::budget::{BudgetError, BudgetTree};

/// Coordinator-level error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordinatorError {
    /// Underlying hierarchical budget rejected the operation.
    Budget(BudgetError),
    /// Resource id is already registered.
    DuplicateResource(String),
    /// Resource id is not registered.
    UnknownResource(String),
    /// Release exceeds bytes committed by this specific resource.
    ReleaseExceedsCommitment {
        /// Resource id.
        resource_id: String,
        /// Requested release bytes.
        requested: u64,
        /// Resource-owned committed bytes.
        committed: u64,
    },
    /// Per-resource accounting overflow.
    AccountingOverflow(String),
}

impl From<BudgetError> for CoordinatorError {
    fn from(value: BudgetError) -> Self {
        Self::Budget(value)
    }
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget(error) => write!(f, "budget error: {error}"),
            Self::DuplicateResource(id) => write!(f, "resource `{id}` is already registered"),
            Self::UnknownResource(id) => write!(f, "resource `{id}` is not registered"),
            Self::ReleaseExceedsCommitment {
                resource_id,
                requested,
                committed,
            } => write!(
                f,
                "resource `{resource_id}` cannot release {requested} bytes; it owns {committed}"
            ),
            Self::AccountingOverflow(id) => {
                write!(f, "resource `{id}` commitment accounting overflow")
            }
        }
    }
}

impl std::error::Error for CoordinatorError {}

/// A registered resource in the coordinator.
#[derive(Clone, Debug)]
pub struct RegisteredResource {
    /// Stable resource identifier.
    pub resource_id: String,
    /// Budget node index in the shared tree.
    pub node: usize,
    /// Priority metadata for higher-level arbitration.
    pub priority: u8,
    /// Bytes committed by this resource through the coordinator.
    pub committed: u64,
}

/// Shared multi-resource coordinator.
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

    /// Shared budget tree.
    pub fn tree(&self) -> &BudgetTree {
        &self.tree
    }

    /// Mutable tree access for topology setup. Callers must finish topology
    /// construction before admitting live resource commitments.
    pub fn tree_mut(&mut self) -> &mut BudgetTree {
        &mut self.tree
    }

    /// Register a unique resource under an existing budget node.
    pub fn register(
        &mut self,
        resource_id: &str,
        node: usize,
        priority: u8,
    ) -> Result<usize, CoordinatorError> {
        self.tree.node(node)?;
        if self.find(resource_id).is_some() {
            return Err(CoordinatorError::DuplicateResource(resource_id.to_string()));
        }
        let index = self.resources.len();
        self.resources.push(RegisteredResource {
            resource_id: resource_id.to_string(),
            node,
            priority,
            committed: 0,
        });
        Ok(index)
    }

    /// Look up a resource by id.
    pub fn find(&self, resource_id: &str) -> Option<&RegisteredResource> {
        self.resources
            .iter()
            .find(|resource| resource.resource_id == resource_id)
    }

    fn find_index(&self, resource_id: &str) -> Result<usize, CoordinatorError> {
        self.resources
            .iter()
            .position(|resource| resource.resource_id == resource_id)
            .ok_or_else(|| CoordinatorError::UnknownResource(resource_id.to_string()))
    }

    /// Registered resources.
    pub fn resources(&self) -> &[RegisteredResource] {
        &self.resources
    }

    /// Bytes currently owned by one registered resource.
    pub fn committed(&self, resource_id: &str) -> Result<u64, CoordinatorError> {
        Ok(self
            .resources
            .get(self.find_index(resource_id)?)
            .expect("index returned from same resource vector")
            .committed)
    }

    /// Admit `bytes` for one resource through the single shared tree.
    ///
    /// Sibling reservations are enforced by [`BudgetTree::try_commit`]. The
    /// `priority` field remains metadata for a future explicit preemption
    /// policy; this coordinator never steals protected reservation implicitly.
    pub fn commit(&mut self, resource_id: &str, bytes: u64) -> Result<(), CoordinatorError> {
        let index = self.find_index(resource_id)?;
        let node = self.resources[index].node;
        let next = self.resources[index]
            .committed
            .checked_add(bytes)
            .ok_or_else(|| CoordinatorError::AccountingOverflow(resource_id.to_string()))?;
        self.tree.try_commit(node, bytes)?;
        self.resources[index].committed = next;
        Ok(())
    }

    /// Release bytes owned by one resource.
    pub fn release(&mut self, resource_id: &str, bytes: u64) -> Result<(), CoordinatorError> {
        let index = self.find_index(resource_id)?;
        let committed = self.resources[index].committed;
        if bytes > committed {
            return Err(CoordinatorError::ReleaseExceedsCommitment {
                resource_id: resource_id.to_string(),
                requested: bytes,
                committed,
            });
        }
        let node = self.resources[index].node;
        self.tree.release(node, bytes)?;
        self.resources[index].committed = committed - bytes;
        Ok(())
    }

    /// Total bytes committed across the shared tree.
    pub fn total_committed(&self) -> u64 {
        self.tree.total_committed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinator() -> ElasticCoordinator {
        let mut coordinator = ElasticCoordinator::new();
        let org = coordinator.tree_mut().add_root(0, 1000, 800, 0);
        let tenant = coordinator
            .tree_mut()
            .add_child(org, 5, 700, 600, 0)
            .unwrap();
        let session = coordinator
            .tree_mut()
            .add_child(tenant, 5, 600, 500, 0)
            .unwrap();
        coordinator.register("ctx", session, 5).unwrap();
        coordinator.register("mem", session, 5).unwrap();
        coordinator
    }

    #[test]
    fn resources_share_one_real_budget() {
        let mut coordinator = coordinator();
        coordinator.commit("ctx", 300).unwrap();
        coordinator.commit("mem", 250).unwrap();
        assert_eq!(coordinator.total_committed(), 550);
        assert_eq!(coordinator.committed("ctx").unwrap(), 300);
        assert_eq!(coordinator.committed("mem").unwrap(), 250);
        assert!(matches!(
            coordinator.commit("mem", 100),
            Err(CoordinatorError::Budget(BudgetError::HardLimitExceeded))
        ));
    }

    #[test]
    fn one_resource_cannot_release_anothers_commitment() {
        let mut coordinator = coordinator();
        coordinator.commit("ctx", 300).unwrap();
        coordinator.commit("mem", 50).unwrap();
        assert!(matches!(
            coordinator.release("mem", 100),
            Err(CoordinatorError::ReleaseExceedsCommitment { .. })
        ));
        assert_eq!(coordinator.total_committed(), 350);
        assert_eq!(coordinator.committed("ctx").unwrap(), 300);
        assert_eq!(coordinator.committed("mem").unwrap(), 50);
    }

    #[test]
    fn release_updates_tree_and_resource_atomically() {
        let mut coordinator = coordinator();
        coordinator.commit("ctx", 300).unwrap();
        coordinator.release("ctx", 125).unwrap();
        assert_eq!(coordinator.committed("ctx").unwrap(), 175);
        assert_eq!(coordinator.total_committed(), 175);
    }

    #[test]
    fn duplicate_and_unknown_ids_fail_closed() {
        let mut coordinator = coordinator();
        let node = coordinator.find("ctx").unwrap().node;
        assert!(matches!(
            coordinator.register("ctx", node, 5),
            Err(CoordinatorError::DuplicateResource(_))
        ));
        assert!(matches!(
            coordinator.commit("nope", 1),
            Err(CoordinatorError::UnknownResource(_))
        ));
    }
}
