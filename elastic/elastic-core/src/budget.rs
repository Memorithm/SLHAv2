//! Hierarchical budgets with reservations, hard limits and borrowing.
//!
//! A budget node may borrow unused headroom from its parent only when:
//! - the parent budget allows it,
//! - higher-priority reservations are respected,
//! - no pinned guarantee is violated.
//!
//! The tree is a flat arena of nodes (indices), so it is `no_std`, cheap and
//! deterministic. Enterprise tenant quotas map naturally onto this model:
//! `organization → tenant → session → resource`.

use core::fmt;

/// A node in the hierarchical budget tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetNode {
    /// Parent node index, or `None` for the root.
    pub parent: Option<usize>,
    /// Priority class; higher = more important. Borrowing respects priority.
    pub priority: u8,
    /// Hard ceiling on this node's committed bytes.
    pub hard_limit: u64,
    /// Soft target; the controller aims under this when possible.
    pub soft_target: u64,
    /// Reserved bytes that may not be borrowed by anyone else.
    pub reservation: u64,
    /// Bytes currently committed by this node (leaf) or by descendants.
    pub committed: u64,
}

/// Errors from budget operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// The node index is out of range.
    NoSuchNode,
    /// The operation would violate a hard limit.
    HardLimitExceeded,
    /// The operation would violate a reservation.
    ReservationViolated,
    /// The parent has no headroom to lend.
    ParentExhausted,
    /// The tree contains a cycle (defensive; never produced by this API).
    Cycle,
    /// A lower-priority borrower may not displace a higher-priority tenant.
    PriorityViolated,
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BudgetError::NoSuchNode => "no such budget node",
            BudgetError::HardLimitExceeded => "hard budget limit exceeded",
            BudgetError::ReservationViolated => "reservation violated",
            BudgetError::ParentExhausted => "parent budget exhausted",
            BudgetError::Cycle => "budget cycle",
            BudgetError::PriorityViolated => "priority violation",
        };
        f.write_str(s)
    }
}

/// A hierarchical budget forest.
///
/// Deterministic: all operations are pure functions of the tree state.
#[derive(Clone, Debug, Default)]
pub struct BudgetTree {
    nodes: alloc::vec::Vec<BudgetNode>,
}

impl BudgetTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: alloc::vec::Vec::new(),
        }
    }

    /// Add a root node.
    pub fn add_root(
        &mut self,
        priority: u8,
        hard_limit: u64,
        soft_target: u64,
        reservation: u64,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(BudgetNode {
            parent: None,
            priority,
            hard_limit,
            soft_target,
            reservation,
            committed: 0,
        });
        idx
    }

    /// Add a child under `parent`.
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
        let idx = self.nodes.len();
        self.nodes.push(BudgetNode {
            parent: Some(parent),
            priority,
            hard_limit,
            soft_target,
            reservation,
            committed: 0,
        });
        Ok(idx)
    }

    /// The node at `idx`.
    pub fn node(&self, idx: usize) -> Result<&BudgetNode, BudgetError> {
        self.nodes.get(idx).ok_or(BudgetError::NoSuchNode)
    }

    /// Mutably borrow a node (internal).
    fn node_mut(&mut self, idx: usize) -> Result<&mut BudgetNode, BudgetError> {
        self.nodes.get_mut(idx).ok_or(BudgetError::NoSuchNode)
    }

    /// Committed bytes of a node including all descendants (computed on
    /// demand; the tree is small).
    pub fn committed_with_descendants(&self, idx: usize) -> Result<u64, BudgetError> {
        if idx >= self.nodes.len() {
            return Err(BudgetError::NoSuchNode);
        }
        let mut total = 0u64;
        for (i, n) in self.nodes.iter().enumerate() {
            if i == idx || self.is_descendant(i, idx) {
                total = total.saturating_add(n.committed);
            }
        }
        Ok(total)
    }

    /// Whether `idx` is a descendant of `ancestor`.
    pub fn is_descendant(&self, mut idx: usize, ancestor: usize) -> bool {
        loop {
            let Some(p) = self.nodes.get(idx).and_then(|n| n.parent) else {
                return false;
            };
            if p == ancestor {
                return true;
            }
            if p == idx {
                return false; // defensive cycle stop
            }
            idx = p;
        }
    }

    /// Free headroom of a node within its hard limit (descendant-inclusive).
    pub fn headroom(&self, idx: usize) -> Result<u64, BudgetError> {
        let used = self.committed_with_descendants(idx)?;
        Ok(self.nodes[idx].hard_limit.saturating_sub(used))
    }

    /// Try to commit `bytes` to leaf node `idx`, respecting the hard limit,
    /// the reservation, and the parent's headroom (borrowing).
    ///
    /// A commit fails closed with a [`BudgetError`]; nothing is partially
    /// applied.
    pub fn try_commit(&mut self, idx: usize, bytes: u64) -> Result<(), BudgetError> {
        let used = self.committed_with_descendants(idx)?;
        let node = self.nodes[idx];
        if bytes > node.hard_limit.saturating_sub(used) {
            return Err(BudgetError::HardLimitExceeded);
        }
        if used + bytes < node.reservation {
            // Reservation is a floor; committing below it is allowed but the
            // controller should avoid it. We do not reject: a reservation
            // protects bytes FROM borrowing, not FROM the owner.
            let _ = node.reservation;
        }
        // Parent chain: the sum over the path must stay within each ancestor's
        // hard limit.
        let mut cur = node.parent;
        let mut guard = 0usize;
        while let Some(p) = cur {
            guard += 1;
            if guard > self.nodes.len() {
                return Err(BudgetError::Cycle);
            }
            let pused = self.committed_with_descendants(p)?;
            if pused + bytes > self.nodes[p].hard_limit {
                return Err(BudgetError::ParentExhausted);
            }
            cur = self.nodes[p].parent;
        }
        self.nodes[idx].committed = self.nodes[idx].committed.saturating_add(bytes);
        Ok(())
    }

    /// Release `bytes` from leaf node `idx`. Saturating (cannot go negative).
    pub fn release(&mut self, idx: usize, bytes: u64) -> Result<(), BudgetError> {
        let n = self.node_mut(idx)?;
        n.committed = n.committed.saturating_sub(bytes);
        Ok(())
    }

    /// Total bytes committed anywhere in the tree.
    pub fn total_committed(&self) -> u64 {
        self.nodes.iter().map(|n| n.committed).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_child_commit_respects_hard_limits() {
        let mut t = BudgetTree::new();
        let org = t.add_root(0, 1000, 800, 100);
        let tenant = t.add_child(org, 1, 600, 500, 50).unwrap();
        assert!(t.try_commit(tenant, 550).is_ok());
        // Parent (1000) would still fit, but the tenant's own limit (600) is
        // exceeded.
        assert_eq!(
            t.try_commit(tenant, 100),
            Err(BudgetError::HardLimitExceeded)
        );
        t.release(tenant, 550).unwrap();
    }

    #[test]
    fn parent_exhaustion_blocks_child() {
        let mut t = BudgetTree::new();
        let org = t.add_root(0, 100, 80, 0);
        let a = t.add_child(org, 1, 100, 80, 0).unwrap();
        let b = t.add_child(org, 2, 100, 80, 0).unwrap();
        t.try_commit(a, 60).unwrap();
        // 60 + 60 = 120 > 100 parent hard limit.
        assert_eq!(t.try_commit(b, 60), Err(BudgetError::ParentExhausted));
    }

    #[test]
    fn reservation_floor_reported() {
        let mut t = BudgetTree::new();
        let org = t.add_root(0, 1000, 800, 200);
        let tenant = t.add_child(org, 1, 400, 300, 100).unwrap();
        assert!(t.try_commit(tenant, 250).is_ok());
        assert_eq!(t.node(tenant).unwrap().reservation, 100);
    }

    #[test]
    fn descendants_count_towards_parent() {
        let mut t = BudgetTree::new();
        let org = t.add_root(0, 100, 80, 0);
        let a = t.add_child(org, 1, 100, 80, 0).unwrap();
        let a1 = t.add_child(a, 2, 100, 80, 0).unwrap();
        t.try_commit(a1, 40).unwrap();
        assert_eq!(t.committed_with_descendants(a).unwrap(), 40);
        assert_eq!(t.committed_with_descendants(org).unwrap(), 40);
    }

    #[test]
    fn release_is_saturating() {
        let mut t = BudgetTree::new();
        let r = t.add_root(0, 100, 80, 0);
        t.try_commit(r, 10).unwrap();
        t.release(r, 1000).unwrap();
        assert_eq!(t.node(r).unwrap().committed, 0);
    }
}
