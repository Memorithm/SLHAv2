//! Hierarchical byte budgets with hard limits and protected reservations.
//!
//! A commit to one subtree must fit both its own hard limit and every ancestor
//! hard limit while preserving the still-unused reservations of sibling
//! subtrees. The structure is deterministic and `no_std` friendly.

use core::fmt;

/// One node in a hierarchical budget tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetNode {
    /// Parent index, or `None` for a root.
    pub parent: Option<usize>,
    /// Coordinator priority metadata; higher values are more important.
    pub priority: u8,
    /// Hard ceiling for this subtree.
    pub hard_limit: u64,
    /// Preferred occupancy target, normalized to `<= hard_limit`.
    pub soft_target: u64,
    /// Protected capacity for this subtree, normalized to `<= hard_limit`.
    pub reservation: u64,
    /// Bytes committed directly by this node.
    pub committed: u64,
}

/// Budget operation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// Node index does not exist.
    NoSuchNode,
    /// The target node's own hard limit would be exceeded.
    HardLimitExceeded,
    /// A sibling reservation would be consumed.
    ReservationViolated,
    /// An ancestor has insufficient total capacity.
    ParentExhausted,
    /// A malformed parent chain contains a cycle.
    Cycle,
    /// Reserved for higher-level coordinators that implement priority-based
    /// preemption. This tree itself never preempts protected reservations.
    PriorityViolated,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NoSuchNode => "no such budget node",
            Self::HardLimitExceeded => "hard budget limit exceeded",
            Self::ReservationViolated => "sibling reservation would be violated",
            Self::ParentExhausted => "parent budget exhausted",
            Self::Cycle => "budget cycle",
            Self::PriorityViolated => "priority violation",
        })
    }
}

/// Hierarchical budget forest backed by a flat deterministic arena.
#[derive(Clone, Debug, Default)]
pub struct BudgetTree {
    nodes: alloc::vec::Vec<BudgetNode>,
}

impl BudgetTree {
    /// Create an empty forest.
    pub fn new() -> Self {
        Self {
            nodes: alloc::vec::Vec::new(),
        }
    }

    /// Add a root. Soft target and reservation are clamped to the hard limit.
    pub fn add_root(
        &mut self,
        priority: u8,
        hard_limit: u64,
        soft_target: u64,
        reservation: u64,
    ) -> usize {
        let index = self.nodes.len();
        self.nodes.push(BudgetNode {
            parent: None,
            priority,
            hard_limit,
            soft_target: soft_target.min(hard_limit),
            reservation: reservation.min(hard_limit),
            committed: 0,
        });
        index
    }

    /// Add a child. Soft target and reservation are clamped to the child's
    /// hard limit.
    pub fn add_child(
        &mut self,
        parent: usize,
        priority: u8,
        hard_limit: u64,
        soft_target: u64,
        reservation: u64,
    ) -> Result<usize, BudgetError> {
        if parent >= self.nodes.len() {
            return Err(BudgetError::NoSuchNode);
        }
        let index = self.nodes.len();
        self.nodes.push(BudgetNode {
            parent: Some(parent),
            priority,
            hard_limit,
            soft_target: soft_target.min(hard_limit),
            reservation: reservation.min(hard_limit),
            committed: 0,
        });
        Ok(index)
    }

    /// Borrow a node by index.
    pub fn node(&self, index: usize) -> Result<&BudgetNode, BudgetError> {
        self.nodes.get(index).ok_or(BudgetError::NoSuchNode)
    }

    fn node_mut(&mut self, index: usize) -> Result<&mut BudgetNode, BudgetError> {
        self.nodes.get_mut(index).ok_or(BudgetError::NoSuchNode)
    }

    /// Bytes committed directly to `index` and all its descendants.
    pub fn committed_with_descendants(&self, index: usize) -> Result<u64, BudgetError> {
        if index >= self.nodes.len() {
            return Err(BudgetError::NoSuchNode);
        }
        let mut total = 0u64;
        for (candidate, node) in self.nodes.iter().enumerate() {
            if candidate == index || self.is_descendant(candidate, index) {
                total = total.saturating_add(node.committed);
            }
        }
        Ok(total)
    }

    /// Whether `index` is a strict descendant of `ancestor`.
    ///
    /// Malformed cycles fail closed as `false` instead of looping forever.
    pub fn is_descendant(&self, mut index: usize, ancestor: usize) -> bool {
        let mut hops = 0usize;
        loop {
            if hops > self.nodes.len() {
                return false;
            }
            let Some(parent) = self.nodes.get(index).and_then(|node| node.parent) else {
                return false;
            };
            if parent == ancestor {
                return true;
            }
            index = parent;
            hops += 1;
        }
    }

    /// Remaining headroom under one node's hard limit, descendants included.
    pub fn headroom(&self, index: usize) -> Result<u64, BudgetError> {
        let used = self.committed_with_descendants(index)?;
        Ok(self.node(index)?.hard_limit.saturating_sub(used))
    }

    /// Transactionally commit bytes to `index`.
    ///
    /// Validation is complete before mutation: own hard limit, every ancestor
    /// hard limit, and all unused immediate-sibling reservations along the
    /// path are checked first. No partial commit is possible.
    pub fn try_commit(&mut self, index: usize, bytes: u64) -> Result<(), BudgetError> {
        if index >= self.nodes.len() {
            return Err(BudgetError::NoSuchNode);
        }
        if bytes == 0 {
            return Ok(());
        }

        let used = self.committed_with_descendants(index)?;
        let node = self.nodes[index];
        if bytes > node.hard_limit.saturating_sub(used) {
            return Err(BudgetError::HardLimitExceeded);
        }

        let mut path_child = index;
        let mut current = node.parent;
        let mut hops = 0usize;
        while let Some(parent) = current {
            hops += 1;
            if hops > self.nodes.len() {
                return Err(BudgetError::Cycle);
            }
            let parent_node = *self.node(parent)?;
            let parent_used = self.committed_with_descendants(parent)?;
            if bytes > parent_node.hard_limit.saturating_sub(parent_used) {
                return Err(BudgetError::ParentExhausted);
            }

            let mut protected = 0u64;
            for (sibling_index, sibling) in self.nodes.iter().enumerate() {
                if sibling.parent == Some(parent) && sibling_index != path_child {
                    let sibling_used = self.committed_with_descendants(sibling_index)?;
                    protected =
                        protected.saturating_add(sibling.reservation.saturating_sub(sibling_used));
                }
            }
            let projected = parent_used.saturating_add(bytes);
            if protected > parent_node.hard_limit.saturating_sub(projected) {
                return Err(BudgetError::ReservationViolated);
            }

            path_child = parent;
            current = parent_node.parent;
        }

        self.nodes[index].committed = self.nodes[index].committed.saturating_add(bytes);
        Ok(())
    }

    /// Release bytes committed directly to `index`; saturates at zero.
    pub fn release(&mut self, index: usize, bytes: u64) -> Result<(), BudgetError> {
        let node = self.node_mut(index)?;
        node.committed = node.committed.saturating_sub(bytes);
        Ok(())
    }

    /// Sum direct commitments without double-counting descendant totals.
    pub fn total_committed(&self) -> u64 {
        self.nodes
            .iter()
            .fold(0u64, |total, node| total.saturating_add(node.committed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_are_normalized() {
        let mut tree = BudgetTree::new();
        let root = tree.add_root(0, 100, 500, 500);
        let node = tree.node(root).unwrap();
        assert_eq!(node.soft_target, 100);
        assert_eq!(node.reservation, 100);
    }

    #[test]
    fn own_and_parent_hard_limits_are_enforced() {
        let mut tree = BudgetTree::new();
        let root = tree.add_root(0, 1000, 800, 0);
        let child = tree.add_child(root, 1, 600, 500, 0).unwrap();
        assert!(tree.try_commit(child, 550).is_ok());
        assert_eq!(
            tree.try_commit(child, 100),
            Err(BudgetError::HardLimitExceeded)
        );

        let other = tree.add_child(root, 1, 1000, 800, 0).unwrap();
        assert_eq!(
            tree.try_commit(other, 500),
            Err(BudgetError::ParentExhausted)
        );
    }

    #[test]
    fn sibling_reservation_is_actually_protected() {
        let mut tree = BudgetTree::new();
        let root = tree.add_root(0, 1000, 800, 0);
        let reserved = tree.add_child(root, 10, 600, 500, 300).unwrap();
        let borrower = tree.add_child(root, 1, 1000, 800, 0).unwrap();
        tree.try_commit(reserved, 100).unwrap();
        assert!(tree.try_commit(borrower, 700).is_ok());
        assert_eq!(
            tree.try_commit(borrower, 1),
            Err(BudgetError::ReservationViolated)
        );
    }

    #[test]
    fn nested_sibling_reservations_are_protected_at_each_ancestor() {
        let mut tree = BudgetTree::new();
        let root = tree.add_root(0, 1000, 800, 0);
        let tenant = tree.add_child(root, 1, 800, 600, 0).unwrap();
        let sibling_tenant = tree.add_child(root, 1, 500, 400, 250).unwrap();
        let session = tree.add_child(tenant, 1, 800, 600, 0).unwrap();
        let sibling_session = tree.add_child(tenant, 1, 400, 300, 200).unwrap();
        tree.try_commit(sibling_tenant, 50).unwrap();
        tree.try_commit(sibling_session, 50).unwrap();

        assert!(tree.try_commit(session, 600).is_ok());
        assert_eq!(
            tree.try_commit(session, 1),
            Err(BudgetError::ReservationViolated)
        );
    }

    #[test]
    fn release_and_headroom_are_saturating() {
        let mut tree = BudgetTree::new();
        let root = tree.add_root(0, 100, 80, 0);
        tree.try_commit(root, 10).unwrap();
        assert_eq!(tree.headroom(root).unwrap(), 90);
        tree.release(root, 1000).unwrap();
        assert_eq!(tree.node(root).unwrap().committed, 0);
        assert_eq!(tree.headroom(root).unwrap(), 100);
    }

    #[test]
    fn descendants_count_toward_parent_without_double_counting_total() {
        let mut tree = BudgetTree::new();
        let root = tree.add_root(0, 100, 80, 0);
        let child = tree.add_child(root, 1, 100, 80, 0).unwrap();
        let leaf = tree.add_child(child, 2, 100, 80, 0).unwrap();
        tree.try_commit(leaf, 40).unwrap();
        assert_eq!(tree.committed_with_descendants(child).unwrap(), 40);
        assert_eq!(tree.committed_with_descendants(root).unwrap(), 40);
        assert_eq!(tree.total_committed(), 40);
    }
}
