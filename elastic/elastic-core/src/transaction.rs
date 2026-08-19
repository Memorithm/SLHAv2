//! Transactional elastic transitions: PREPARE → VALIDATE → COMMIT → RELEASE,
//! with ROLLBACK on every failed mutating phase.
//!
//! A transition must never destroy the old representation, fail while building
//! the replacement, and then continue with partially mutated state. The
//! executor therefore asks `rollback()` immediately after any failed phase,
//! including PREPARE, and reports rollback failure as a hard error.

/// Outcome of one transition phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseOutcome {
    /// The phase succeeded.
    Ok,
    /// The phase failed.
    Failed,
}

/// Transactional representation transition protocol.
pub trait Transaction {
    /// State being transitioned.
    type State;

    /// Build the candidate representation while preserving the currently live
    /// representation. Implementations may allocate scratch state.
    fn prepare(&mut self, state: &mut Self::State) -> PhaseOutcome;

    /// Validate the prepared representation without committing it.
    fn validate(&mut self, state: &mut Self::State) -> PhaseOutcome;

    /// Atomically switch to the prepared representation. If this reports
    /// failure, [`Self::rollback`] must be able to restore the pre-transition
    /// state.
    fn commit(&mut self, state: &mut Self::State) -> PhaseOutcome;

    /// Restore the complete pre-transition state and release any scratch built
    /// by PREPARE. The return value describes *this rollback call*, avoiding a
    /// stale success flag when transaction objects are reused.
    fn rollback(&mut self, state: &mut Self::State) -> PhaseOutcome;

    /// Release the old representation after a successful commit. This phase is
    /// intentionally infallible at the protocol level: implementations should
    /// make release idempotent/best-effort and retain enough ownership to clean
    /// up later if a platform deallocation reports an error.
    fn release(&mut self, _state: &mut Self::State) {}
}

/// Compatibility marker for code written against the first Elastic prototype.
///
/// Rollback success is now returned directly by [`Transaction::rollback`], so
/// no separate sticky status method is needed. Every `Transaction`
/// automatically satisfies this marker; it remains exported to avoid breaking
/// controller/facade code while the API is still pre-release.
pub trait RollbackAware: Transaction {}
impl<T: Transaction + ?Sized> RollbackAware for T {}

/// Result of running a transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionOutcome {
    /// Commit and release completed.
    Committed,
    /// Prepare failed and rollback restored the pre-transition state.
    PrepareFailed,
    /// Validation failed and rollback restored the pre-transition state.
    ValidationFailed,
    /// Commit failed and rollback restored the pre-transition state.
    CommitFailed,
    /// Rollback itself failed; the caller must treat state as potentially
    /// inconsistent and fail closed.
    RollbackFailed,
}

/// Run one deterministic prepare/validate/commit transition.
pub fn run_transaction<T: Transaction>(
    tx: &mut T,
    state: &mut T::State,
) -> TransitionOutcome {
    if tx.prepare(state) == PhaseOutcome::Failed {
        return rollback_or(tx, state, TransitionOutcome::PrepareFailed);
    }
    if tx.validate(state) == PhaseOutcome::Failed {
        return rollback_or(tx, state, TransitionOutcome::ValidationFailed);
    }
    if tx.commit(state) == PhaseOutcome::Failed {
        return rollback_or(tx, state, TransitionOutcome::CommitFailed);
    }
    tx.release(state);
    TransitionOutcome::Committed
}

fn rollback_or<T: Transaction>(
    tx: &mut T,
    state: &mut T::State,
    failure: TransitionOutcome,
) -> TransitionOutcome {
    if tx.rollback(state) == PhaseOutcome::Ok {
        failure
    } else {
        TransitionOutcome::RollbackFailed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct RepState {
        live: u8,
        scratch: Option<u8>,
        released: bool,
    }

    struct FlipTx {
        fail_prepare_after_scratch: bool,
        fail_validate: bool,
        fail_commit: bool,
        fail_rollback: bool,
        rollback_calls: usize,
    }

    impl Transaction for FlipTx {
        type State = RepState;

        fn prepare(&mut self, state: &mut RepState) -> PhaseOutcome {
            state.scratch = Some(state.live ^ 0xFF);
            if self.fail_prepare_after_scratch {
                PhaseOutcome::Failed
            } else {
                PhaseOutcome::Ok
            }
        }

        fn validate(&mut self, state: &mut RepState) -> PhaseOutcome {
            if self.fail_validate || state.scratch.is_none() {
                PhaseOutcome::Failed
            } else {
                PhaseOutcome::Ok
            }
        }

        fn commit(&mut self, state: &mut RepState) -> PhaseOutcome {
            if self.fail_commit {
                return PhaseOutcome::Failed;
            }
            state.live = state.scratch.take().expect("validated scratch");
            PhaseOutcome::Ok
        }

        fn rollback(&mut self, state: &mut RepState) -> PhaseOutcome {
            self.rollback_calls += 1;
            if self.fail_rollback {
                return PhaseOutcome::Failed;
            }
            state.scratch = None;
            PhaseOutcome::Ok
        }

        fn release(&mut self, state: &mut RepState) {
            state.released = true;
        }
    }

    fn state() -> RepState {
        RepState {
            live: 1,
            scratch: None,
            released: false,
        }
    }

    fn tx() -> FlipTx {
        FlipTx {
            fail_prepare_after_scratch: false,
            fail_validate: false,
            fail_commit: false,
            fail_rollback: false,
            rollback_calls: 0,
        }
    }

    #[test]
    fn successful_commit_switches_and_releases() {
        let mut state = state();
        let mut tx = tx();
        assert_eq!(
            run_transaction(&mut tx, &mut state),
            TransitionOutcome::Committed
        );
        assert_eq!(state.live, 0xFE);
        assert!(state.released);
        assert_eq!(tx.rollback_calls, 0);
    }

    #[test]
    fn failed_prepare_rolls_back_partial_scratch() {
        let mut state = state();
        let mut tx = tx();
        tx.fail_prepare_after_scratch = true;
        assert_eq!(
            run_transaction(&mut tx, &mut state),
            TransitionOutcome::PrepareFailed
        );
        assert_eq!(state.live, 1);
        assert!(state.scratch.is_none());
        assert!(!state.released);
        assert_eq!(tx.rollback_calls, 1);
    }

    #[test]
    fn failed_validate_rolls_back_and_preserves_live() {
        let mut state = state();
        let mut tx = tx();
        tx.fail_validate = true;
        assert_eq!(
            run_transaction(&mut tx, &mut state),
            TransitionOutcome::ValidationFailed
        );
        assert_eq!(state.live, 1);
        assert!(state.scratch.is_none());
        assert_eq!(tx.rollback_calls, 1);
    }

    #[test]
    fn failed_commit_rolls_back() {
        let mut state = state();
        let mut tx = tx();
        tx.fail_commit = true;
        assert_eq!(
            run_transaction(&mut tx, &mut state),
            TransitionOutcome::CommitFailed
        );
        assert_eq!(state.live, 1);
        assert!(state.scratch.is_none());
        assert_eq!(tx.rollback_calls, 1);
    }

    #[test]
    fn rollback_failure_is_hard_error_even_on_prepare_failure() {
        let mut state = state();
        let mut tx = tx();
        tx.fail_prepare_after_scratch = true;
        tx.fail_rollback = true;
        assert_eq!(
            run_transaction(&mut tx, &mut state),
            TransitionOutcome::RollbackFailed
        );
        assert_eq!(tx.rollback_calls, 1);
        assert!(state.scratch.is_some());
    }

    #[test]
    fn reused_transaction_cannot_inherit_stale_rollback_success() {
        let mut tx = tx();
        let mut first = state();
        tx.fail_validate = true;
        assert_eq!(
            run_transaction(&mut tx, &mut first),
            TransitionOutcome::ValidationFailed
        );

        let mut second = state();
        tx.fail_rollback = true;
        assert_eq!(
            run_transaction(&mut tx, &mut second),
            TransitionOutcome::RollbackFailed
        );
        assert_eq!(tx.rollback_calls, 2);
    }
}
