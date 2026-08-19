//! Validated tier state machines (HOT/WARM/COLD/…).
//!
//! A machine is a directed graph over exact `(name, rank)` tier descriptors.
//! Runtime construction uses [`TierMachine::try_new`]; [`TierMachine::new`]
//! is the fail-fast convenience constructor for statically authored graphs.

use core::fmt;
use core::marker::PhantomData;

/// A tier of an elastic resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tier {
    /// Stable tier identifier.
    pub name: &'static str,
    /// Sortable rank; higher = more resident/faster representation.
    pub rank: u8,
}

/// A directed transition between two exact tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierTransition {
    /// Source tier.
    pub from: Tier,
    /// Destination tier.
    pub to: Tier,
}

/// Tier graph/transition error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierError {
    /// The transition is not declared in the table.
    TransitionNotAllowed {
        /// Source tier name.
        from: &'static str,
        /// Destination tier name.
        to: &'static str,
    },
    /// A tier name is not part of this machine.
    UnknownTier(&'static str),
    /// The name exists, but the supplied rank does not match the canonical
    /// descriptor registered by this machine.
    TierDescriptorMismatch {
        /// Tier name.
        name: &'static str,
        /// Canonical rank.
        expected_rank: u8,
        /// Supplied rank.
        actual_rank: u8,
    },
    /// Two declared tiers use the same stable name.
    DuplicateTier(&'static str),
    /// The same directed edge was declared twice.
    DuplicateTransition {
        /// Source tier name.
        from: &'static str,
        /// Destination tier name.
        to: &'static str,
    },
}

impl fmt::Display for TierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TransitionNotAllowed { from, to } => {
                write!(f, "tier transition not allowed: {from} -> {to}")
            }
            Self::UnknownTier(tier) => write!(f, "unknown tier: {tier}"),
            Self::TierDescriptorMismatch {
                name,
                expected_rank,
                actual_rank,
            } => write!(
                f,
                "tier descriptor mismatch for {name}: rank {actual_rank}, expected {expected_rank}"
            ),
            Self::DuplicateTier(tier) => write!(f, "duplicate tier: {tier}"),
            Self::DuplicateTransition { from, to } => {
                write!(f, "duplicate tier transition: {from} -> {to}")
            }
        }
    }
}

/// A validated tier state machine.
#[derive(Clone, Debug)]
pub struct TierMachine {
    tiers: alloc::vec::Vec<Tier>,
    transitions: alloc::vec::Vec<TierTransition>,
}

impl TierMachine {
    /// Build a machine and panic if the static graph is invalid.
    ///
    /// Prefer [`Self::try_new`] when graph data comes from runtime/config input.
    pub fn new(tiers: alloc::vec::Vec<Tier>, transitions: alloc::vec::Vec<TierTransition>) -> Self {
        Self::try_new(tiers, transitions).expect("invalid static tier graph")
    }

    /// Validate and build a tier machine.
    pub fn try_new(
        tiers: alloc::vec::Vec<Tier>,
        transitions: alloc::vec::Vec<TierTransition>,
    ) -> Result<Self, TierError> {
        for (index, tier) in tiers.iter().enumerate() {
            if tiers[..index]
                .iter()
                .any(|candidate| candidate.name == tier.name)
            {
                return Err(TierError::DuplicateTier(tier.name));
            }
        }

        for (index, transition) in transitions.iter().enumerate() {
            validate_descriptor(&tiers, transition.from)?;
            validate_descriptor(&tiers, transition.to)?;
            if transitions[..index].contains(transition) {
                return Err(TierError::DuplicateTransition {
                    from: transition.from.name,
                    to: transition.to.name,
                });
            }
        }

        Ok(Self { tiers, transitions })
    }

    /// All declared tiers.
    pub fn tiers(&self) -> &[Tier] {
        &self.tiers
    }

    /// All declared transitions.
    pub fn transitions(&self) -> &[TierTransition] {
        &self.transitions
    }

    /// Look up a tier by stable name.
    pub fn find(&self, name: &'static str) -> Option<Tier> {
        self.tiers.iter().copied().find(|tier| tier.name == name)
    }

    /// Validate a directed move between exact canonical tier descriptors.
    pub fn transition(&self, from: Tier, to: Tier) -> Result<(), TierError> {
        validate_descriptor(&self.tiers, from)?;
        validate_descriptor(&self.tiers, to)?;
        if self
            .transitions
            .iter()
            .any(|transition| transition.from == from && transition.to == to)
        {
            Ok(())
        } else {
            Err(TierError::TransitionNotAllowed {
                from: from.name,
                to: to.name,
            })
        }
    }
}

fn validate_descriptor(tiers: &[Tier], tier: Tier) -> Result<(), TierError> {
    let Some(canonical) = tiers.iter().copied().find(|candidate| candidate.name == tier.name) else {
        return Err(TierError::UnknownTier(tier.name));
    };
    if canonical.rank != tier.rank {
        return Err(TierError::TierDescriptorMismatch {
            name: tier.name,
            expected_rank: canonical.rank,
            actual_rank: tier.rank,
        });
    }
    Ok(())
}

impl fmt::Display for TierMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TierMachine({} tiers, {} transitions)",
            self.tiers.len(),
            self.transitions.len()
        )
    }
}

/// Typed tier state holder with a validated machine.
#[derive(Clone, Copy, Debug)]
pub struct TierState<T> {
    current: T,
    machine: &'static TierMachine,
    _marker: PhantomData<T>,
}

impl<T: Copy + Eq + fmt::Debug> PartialEq for TierState<T> {
    fn eq(&self, other: &Self) -> bool {
        self.current == other.current
    }
}

impl<T: Copy + Eq + fmt::Debug> Eq for TierState<T> {}

impl<T: Copy + Eq + fmt::Debug> TierState<T> {
    /// Construct a typed state after validating its initial descriptor against
    /// the machine.
    pub fn new(current: T, machine: &'static TierMachine) -> Result<Self, TierError>
    where
        T: TierLike,
    {
        validate_descriptor(machine.tiers(), current.as_tier())?;
        Ok(Self {
            current,
            machine,
            _marker: PhantomData,
        })
    }

    /// Current tier.
    pub fn current(&self) -> T {
        self.current
    }

    /// Try to move to `to`, validating against the machine.
    pub fn try_move(&mut self, to: T) -> Result<(), TierError>
    where
        T: TierLike,
    {
        self.machine
            .transition(self.current.as_tier(), to.as_tier())?;
        self.current = to;
        Ok(())
    }
}

/// Bridge between a user tier enum and the core [`Tier`] representation.
pub trait TierLike: Copy {
    /// The core tier descriptor for this variant.
    fn as_tier(self) -> Tier;
    /// Rebuild a variant from a canonical core tier.
    fn from_tier(tier: Tier) -> Self;
}

/// Build a validated machine from `(name, rank)` and `(from, to)` lists.
pub fn machine_from_lists(
    tiers: &[(&'static str, u8)],
    transitions: &[(&'static str, &'static str)],
) -> Result<TierMachine, TierError> {
    let tier_list: alloc::vec::Vec<Tier> = tiers
        .iter()
        .map(|&(name, rank)| Tier { name, rank })
        .collect();
    let mut transition_list = alloc::vec::Vec::with_capacity(transitions.len());
    for &(from, to) in transitions {
        let from_tier = tier_list
            .iter()
            .copied()
            .find(|tier| tier.name == from)
            .ok_or(TierError::UnknownTier(from))?;
        let to_tier = tier_list
            .iter()
            .copied()
            .find(|tier| tier.name == to)
            .ok_or(TierError::UnknownTier(to))?;
        transition_list.push(TierTransition {
            from: from_tier,
            to: to_tier,
        });
    }
    TierMachine::try_new(tier_list, transition_list)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_machine() -> TierMachine {
        machine_from_lists(
            &[
                ("pinned", 5),
                ("hot", 4),
                ("warm", 3),
                ("cold", 2),
                ("evicted", 1),
            ],
            &[
                ("hot", "warm"),
                ("warm", "hot"),
                ("warm", "cold"),
                ("cold", "warm"),
                ("cold", "evicted"),
                ("evicted", "cold"),
                ("pinned", "hot"),
                ("hot", "pinned"),
            ],
        )
        .unwrap()
    }

    #[test]
    fn declared_and_undeclared_transitions_are_distinguished() {
        let machine = test_machine();
        let hot = machine.find("hot").unwrap();
        let warm = machine.find("warm").unwrap();
        let evicted = machine.find("evicted").unwrap();
        assert!(machine.transition(hot, warm).is_ok());
        assert_eq!(
            machine.transition(hot, evicted),
            Err(TierError::TransitionNotAllowed {
                from: "hot",
                to: "evicted"
            })
        );
    }

    #[test]
    fn unknown_and_mismatched_descriptors_fail_closed() {
        let machine = test_machine();
        let hot = machine.find("hot").unwrap();
        assert_eq!(
            machine.transition(
                hot,
                Tier {
                    name: "nope",
                    rank: 9
                }
            ),
            Err(TierError::UnknownTier("nope"))
        );
        assert_eq!(
            machine.transition(
                Tier {
                    name: "hot",
                    rank: 99
                },
                machine.find("warm").unwrap()
            ),
            Err(TierError::TierDescriptorMismatch {
                name: "hot",
                expected_rank: 4,
                actual_rank: 99,
            })
        );
    }

    #[test]
    fn constructor_rejects_duplicate_tiers_and_edges() {
        assert_eq!(
            TierMachine::try_new(
                alloc::vec![
                    Tier { name: "hot", rank: 2 },
                    Tier { name: "hot", rank: 1 }
                ],
                alloc::vec![]
            ),
            Err(TierError::DuplicateTier("hot"))
        );

        let hot = Tier { name: "hot", rank: 2 };
        let warm = Tier { name: "warm", rank: 1 };
        let edge = TierTransition {
            from: hot,
            to: warm,
        };
        assert_eq!(
            TierMachine::try_new(alloc::vec![hot, warm], alloc::vec![edge, edge]),
            Err(TierError::DuplicateTransition {
                from: "hot",
                to: "warm"
            })
        );
    }

    #[test]
    fn list_builder_rejects_unknown_endpoints() {
        assert_eq!(
            machine_from_lists(&[("hot", 2)], &[("hot", "cold")]),
            Err(TierError::UnknownTier("cold"))
        );
    }

    #[test]
    fn pinned_is_protected_from_eviction() {
        let machine = test_machine();
        let pinned = machine.find("pinned").unwrap();
        let evicted = machine.find("evicted").unwrap();
        assert!(machine.transition(pinned, evicted).is_err());
    }
}
