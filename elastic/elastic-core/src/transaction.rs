//! Transactional elastic transitions: PREPARE → VALIDATE → COMMIT → RELEASE,
//! with ROLLBACK on failure.
//!
//! The failure that must never happen: destroy the raw representation, attempt
//! compression, compression fails, continue with a corrupted cache. Every
//! transition goes through a transactional executor that only commits after
//! validation, and rolls back on any failure.

/// Outcome of a transactional phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseOutcome {
    /// The phase succeeded.
    Ok,
    /// The phase failed; rollback is required (if a prepare/validate already
    /// ran) or the transition is aborted cleanly.
    Failed,
}

/// The transactional transition protocol.
///
/// Implementations must be **idempotent where practical** and must never
/// destroy the old representation before commit. The executor calls
/// [`Transaction::prepare`], then [`Transaction::validate`], then
/// [`Transaction::commit`], and on any failure [`Transaction::rollback`].
pub trait Transaction {
    /// The type of the state being transitioned.
    type State;

    /// Build the new representation without touching the old one.
    /// Failure here must leave the old state fully intact.
    fn prepare(&mut self, state: &mut Self::State) -> PhaseOutcome;

    /// Validate the new representation before commit. Failure aborts the
    /// transition with the old state intact.
    fn validate(&mut self, state: &mut Self::State) -> PhaseOutcome;

    /// Atomically switch the state pointer/representation. After this
    /// returns `Ok`, the old representation may be released.
    fn commit(&mut self, state: &mut Self::State) -> PhaseOutcome;

    /// Restore the old representation. Called when any phase before commit
    /// fails, and also when commit itself fails (if commit can fail).
    fn rollback(&mut self, state: &mut Self::State);

    /// Release the old representation after a successful commit. Best-effort.
    fn release(&mut self, _state: &mut Self::State) {}
}

/// The result of running a transactional transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionOutcome {
    /// The transition committed successfully.
    Committed,
    /// Prepare failed; nothing changed.
    PrepareFailed,
    /// Validation failed; rolled back, nothing changed.
    ValidationFailed,
    /// Commit failed; rolled back, nothing changed.
    CommitFailed,
    /// Rollback itself failed (the state may be inconsistent; the caller
    /// must treat this as a hard error).
    RollbackFailed,
}

/// Run a transactional transition. This is the single entry point the
/// controller uses; it is deterministic and never panics on phase failure.
pub fn run_transaction<T: Transaction + RollbackAware>(
    tx: &mut T,
    state: &mut T::State,
) -> TransitionOutcome {
    if tx.prepare(state) == PhaseOutcome::Failed {
        return TransitionOutcome::PrepareFailed;
    }
    if tx.validate(state) == PhaseOutcome::Failed {
        tx.rollback(state);
        if tx.rolled_back_ok() {
            TransitionOutcome::ValidationFailed
        } else {
            TransitionOutcome::RollbackFailed
        }
    } else if tx.commit(state) == PhaseOutcome::Failed {
        tx.rollback(state);
        if tx.rolled_back_ok() {
            TransitionOutcome::CommitFailed
        } else {
            TransitionOutcome::RollbackFailed
        }
    } else {
        tx.release(state);
        TransitionOutcome::Committed
    }
}

/// Extension of [`Transaction`] that lets the executor distinguish a
/// successful rollback from a failed one.
pub trait RollbackAware: Transaction {
    /// Whether the last [`Transaction::rollback`] call restored the old state.
    fn rolled_back_ok(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple two-representation state for testing.
    #[derive(Debug, Clone, PartialEq)]
    struct RepState {
        /// The "live" representation id.
        live: u8,
        /// A scratch representation the transaction builds.
        scratch: Option<u8>,
        /// Whether the old representation was released.
        released: bool,
    }

    struct FlipTx {
        fail_prepare: bool,
        fail_validate: bool,
        fail_commit: bool,
        fail_rollback: bool,
        rolled_back: bool,
    }

    impl Transaction for FlipTx {
        type State = RepState;

        fn prepare(&mut self, state: &mut RepState) -> PhaseOutcome {
            if self.fail_prepare {
                return PhaseOutcome::Failed;
            }
            state.scratch = Some(state.live ^ 0xFF);
            PhaseOutcome::Ok
        }

        fn validate(&mut self, state: &mut RepState) -> PhaseOutcome {
            if self.fail_validate {
                return PhaseOutcome::Failed;
            }
            if state.scratch.is_none() {
                return PhaseOutcome::Failed;
            }
            PhaseOutcome::Ok
        }

        fn commit(&mut self, state: &mut RepState) -> PhaseOutcome {
            if self.fail_commit {
                return PhaseOutcome::Failed;
            }
            state.live = state.scratch.take().expect("validated scratch");
            PhaseOutcome::Ok
        }

        fn rollback(&mut self, state: &mut RepState) {
            if self.fail_rollback {
                return;
            }
            state.scratch = None; // old `live` was never touched
            self.rolled_back = true;
        }

        fn release(&mut self, state: &mut RepState) {
            state.released = true;
        }
    }

    impl RollbackAware for FlipTx {
        fn rolled_back_ok(&self) -> bool {
            self.rolled_back
        }
    }

    #[test]
    fn successful_commit_switches_and_releases() {
        let mut s = RepState {
            live: 1,
            scratch: None,
            released: false,
        };
        let mut tx = FlipTx {
            fail_prepare: false,
            fail_validate: false,
            fail_commit: false,
            fail_rollback: false,
            rolled_back: false,
        };
        assert_eq!(
            run_transaction(&mut tx, &mut s),
            TransitionOutcome::Committed
        );
        assert_eq!(s.live, 0xFE);
        assert!(s.released);
        assert!(!tx.rolled_back);
    }

    #[test]
    fn failed_prepare_leaves_old_state_intact() {
        let mut s = RepState {
            live: 1,
            scratch: None,
            released: false,
        };
        let mut tx = FlipTx {
            fail_prepare: true,
            fail_validate: false,
            fail_commit: false,
            fail_rollback: false,
            rolled_back: false,
        };
        assert_eq!(
            run_transaction(&mut tx, &mut s),
            TransitionOutcome::PrepareFailed
        );
        assert_eq!(s.live, 1);
        assert!(s.scratch.is_none());
        assert!(!s.released);
    }

    #[test]
    fn failed_validate_rolls_back_and_preserves_live() {
        let mut s = RepState {
            live: 1,
            scratch: None,
            released: false,
        };
        let mut tx = FlipTx {
            fail_prepare: false,
            fail_validate: true,
            fail_commit: false,
            fail_rollback: false,
            rolled_back: false,
        };
        assert_eq!(
            run_transaction(&mut tx, &mut s),
            TransitionOutcome::ValidationFailed
        );
        assert_eq!(s.live, 1);
        assert!(s.scratch.is_none());
        assert!(tx.rolled_back);
    }

    #[test]
    fn failed_commit_rolls_back() {
        let mut s = RepState {
            live: 1,
            scratch: None,
            released: false,
        };
        let mut tx = FlipTx {
            fail_prepare: false,
            fail_validate: false,
            fail_commit: true,
            fail_rollback: false,
            rolled_back: false,
        };
        assert_eq!(
            run_transaction(&mut tx, &mut s),
            TransitionOutcome::CommitFailed
        );
        assert_eq!(s.live, 1);
        assert!(tx.rolled_back);
    }

    #[test]
    fn rollback_failure_is_reported_as_hard_error() {
        let mut s = RepState {
            live: 1,
            scratch: None,
            released: false,
        };
        let mut tx = FlipTx {
            fail_prepare: false,
            fail_validate: true,
            fail_commit: false,
            fail_rollback: true,
            rolled_back: false,
        };
        // A failed rollback still reports RollbackFailed.
        assert_eq!(
            run_transaction(&mut tx, &mut s),
            TransitionOutcome::RollbackFailed
        );
        // The old live value is still intact (commit never ran).
        assert_eq!(s.live, 1);
    }
}
