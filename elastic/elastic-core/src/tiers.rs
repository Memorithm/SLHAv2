//! Tier state machines (HOT/WARM/COLD/…) with validated transition tables.
//!
//! Tiers are declared as a directed graph of allowed transitions. Invalid
//! transitions are rejected at runtime with a clear error; the
//! `elastic-macros` crate can additionally reject statically-declared
//! invalid transitions at compile time where the graph is known.

use core::fmt;
use core::marker::PhantomData;

/// A tier of an elastic resource (HOT, WARM, COLD, EVICTED, PINNED, …).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tier {
    /// Stable tier identifier.
    pub name: &'static str,
    /// Sortable rank; higher = more resident/faster representation.
    pub rank: u8,
}

/// A directed transition between two tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierTransition {
    /// From tier.
    pub from: Tier,
    /// To tier.
    pub to: Tier,
}

/// Errors from tier state transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierError {
    /// The transition is not declared in the table.
    TransitionNotAllowed {
        /// Source tier name.
        from: &'static str,
        /// Destination tier name.
        to: &'static str,
    },
    /// The tier is not part of this table.
    UnknownTier(&'static str),
}

impl fmt::Display for TierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TierError::TransitionNotAllowed { from, to } => {
                write!(f, "tier transition not allowed: {from} -> {to}")
            }
            TierError::UnknownTier(t) => write!(f, "unknown tier: {t}"),
        }
    }
}

/// A validated tier state machine.
///
/// Built from an explicit transition table; `transition(from, to)` returns
/// `Ok(())` only for declared edges. Deterministic and allocation-free.
#[derive(Clone, Debug)]
pub struct TierMachine {
    tiers: alloc::vec::Vec<Tier>,
    transitions: alloc::vec::Vec<TierTransition>,
}

impl TierMachine {
    /// Build a machine from an explicit tier list and transition table.
    pub fn new(tiers: alloc::vec::Vec<Tier>, transitions: alloc::vec::Vec<TierTransition>) -> Self {
        Self { tiers, transitions }
    }

    /// All declared tiers.
    pub fn tiers(&self) -> &[Tier] {
        &self.tiers
    }

    /// Look up a tier by name.
    pub fn find(&self, name: &'static str) -> Option<Tier> {
        self.tiers.iter().copied().find(|t| t.name == name)
    }

    /// Validate a transition against the table.
    pub fn transition(&self, from: Tier, to: Tier) -> Result<(), TierError> {
        if self.find(from.name).is_none() || self.find(to.name).is_none() {
            return Err(TierError::UnknownTier(if self.find(from.name).is_none() {
                from.name
            } else {
                to.name
            }));
        }
        if self
            .transitions
            .iter()
            .any(|t| t.from == from && t.to == to)
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

impl core::fmt::Display for TierMachine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TierMachine({} tiers, {} transitions)",
            self.tiers.len(),
            self.transitions.len()
        )
    }
}

/// A typed tier state holder with a validated machine.
///
/// `T` is the tier enum; `TIER` is its name, `RANK` its rank. The pair is
/// usually produced by `elastic_macros::tier_enum!` or hand-written.
#[derive(Clone, Copy, Debug)]
pub struct TierState<T> {
    current: T,
    machine: &'static TierMachine,
    _marker: PhantomData<T>,
}

impl<T: Copy + Eq + core::fmt::Debug> PartialEq for TierState<T> {
    fn eq(&self, other: &Self) -> bool {
        self.current == other.current
    }
}
impl<T: Copy + Eq + core::fmt::Debug> Eq for TierState<T> {}

impl<T: Copy + Eq + fmt::Debug> TierState<T> {
    /// Current tier.
    pub fn current(&self) -> T {
        self.current
    }

    /// Try to move to `to`, validating against the machine.
    pub fn try_move(&mut self, to: T) -> Result<(), TierError>
    where
        T: TierLike,
    {
        let from = self.current;
        let from_tier = from.as_tier();
        let to_tier = to.as_tier();
        self.machine.transition(from_tier, to_tier)?;
        self.current = to;
        Ok(())
    }
}

/// Bridge between a user tier enum and the core [`Tier`] representation.
pub trait TierLike: Copy {
    /// The core tier descriptor (name + rank) for this variant.
    fn as_tier(self) -> Tier;
    /// Rebuild a variant from a core tier (infallible by construction when
    /// the enum and the machine share their tier set).
    fn from_tier(t: Tier) -> Self;
}

/// Build a `TierMachine` from a const list of `(name, rank)` pairs and a
/// const list of `(from, to)` name pairs.
///
/// Note: `TierMachine` contains `Vec`s, so this cannot be a `const fn`
/// returning a usable value. Prefer constructing `TierMachine` directly with
/// `alloc::vec!` in a `const` item (as the tests do), or use the
/// `elastic_macros::elastic_state!` macro which generates validated tables.
pub fn machine_from_lists(
    tiers: &[(&'static str, u8)],
    transitions: &[(&'static str, &'static str)],
) -> TierMachine {
    let mut tier_list = alloc::vec::Vec::new();
    for &(name, rank) in tiers {
        tier_list.push(Tier { name, rank });
    }
    let mut trans_list = alloc::vec::Vec::new();
    for &(from, to) in transitions {
        let f = tier_list
            .iter()
            .copied()
            .find(|t| t.name == from)
            .unwrap_or(Tier {
                name: from,
                rank: 0,
            });
        let t = tier_list
            .iter()
            .copied()
            .find(|t| t.name == to)
            .unwrap_or(Tier { name: to, rank: 0 });
        trans_list.push(TierTransition { from: f, to: t });
    }
    TierMachine::new(tier_list, trans_list)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_machine() -> TierMachine {
        TierMachine::new(
            alloc::vec![
                Tier {
                    name: "pinned",
                    rank: 5
                },
                Tier {
                    name: "hot",
                    rank: 4
                },
                Tier {
                    name: "warm",
                    rank: 3
                },
                Tier {
                    name: "cold",
                    rank: 2
                },
                Tier {
                    name: "evicted",
                    rank: 1
                },
            ],
            alloc::vec![
                TierTransition {
                    from: Tier {
                        name: "hot",
                        rank: 4
                    },
                    to: Tier {
                        name: "warm",
                        rank: 3
                    }
                },
                TierTransition {
                    from: Tier {
                        name: "warm",
                        rank: 3
                    },
                    to: Tier {
                        name: "hot",
                        rank: 4
                    }
                },
                TierTransition {
                    from: Tier {
                        name: "warm",
                        rank: 3
                    },
                    to: Tier {
                        name: "cold",
                        rank: 2
                    }
                },
                TierTransition {
                    from: Tier {
                        name: "cold",
                        rank: 2
                    },
                    to: Tier {
                        name: "warm",
                        rank: 3
                    }
                },
                TierTransition {
                    from: Tier {
                        name: "cold",
                        rank: 2
                    },
                    to: Tier {
                        name: "evicted",
                        rank: 1
                    }
                },
                TierTransition {
                    from: Tier {
                        name: "evicted",
                        rank: 1
                    },
                    to: Tier {
                        name: "cold",
                        rank: 2
                    }
                },
                TierTransition {
                    from: Tier {
                        name: "pinned",
                        rank: 5
                    },
                    to: Tier {
                        name: "hot",
                        rank: 4
                    }
                },
                TierTransition {
                    from: Tier {
                        name: "hot",
                        rank: 4
                    },
                    to: Tier {
                        name: "pinned",
                        rank: 5
                    }
                },
            ],
        )
    }

    #[test]
    fn declared_transitions_pass() {
        let m = test_machine();
        let hot = m.find("hot").unwrap();
        let warm = m.find("warm").unwrap();
        assert!(m.transition(hot, warm).is_ok());
        assert!(m.transition(warm, hot).is_ok());
    }

    #[test]
    fn undeclared_transition_rejected() {
        let m = test_machine();
        let hot = m.find("hot").unwrap();
        let evicted = m.find("evicted").unwrap();
        assert_eq!(
            m.transition(hot, evicted),
            Err(TierError::TransitionNotAllowed {
                from: "hot",
                to: "evicted"
            })
        );
    }

    #[test]
    fn unknown_tier_rejected() {
        let m = test_machine();
        let hot = m.find("hot").unwrap();
        assert_eq!(
            m.transition(
                hot,
                Tier {
                    name: "nope",
                    rank: 9
                }
            ),
            Err(TierError::UnknownTier("nope"))
        );
    }

    #[test]
    fn pinned_is_protected_from_eviction() {
        let m = test_machine();
        let pinned = m.find("pinned").unwrap();
        let evicted = m.find("evicted").unwrap();
        assert!(m.transition(pinned, evicted).is_err());
    }
}
