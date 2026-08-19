//! The adaptive-value concept: `Fixed`, `Auto`, `Adaptive { min, max }`,
//! `Pinned`.
//!
//! `Auto` means the ECA owns the decision **continuously** — it never means
//! "choose a default once". `Pinned` protects a value from adaptation that
//! would violate pinning semantics.

pub use crate::ElasticValue;

/// A typed holder that tracks the currently-assigned value of an adaptive
/// parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveValue<T: Copy + PartialOrd> {
    contract: ElasticValue<T>,
    assigned: Option<T>,
}

impl<T: Copy + PartialOrd> AdaptiveValue<T> {
    /// Create from a value contract.
    pub const fn new(contract: ElasticValue<T>) -> Self {
        Self {
            contract,
            assigned: None,
        }
    }

    /// The value contract.
    pub const fn contract(&self) -> &ElasticValue<T> {
        &self.contract
    }

    /// The currently effective value, if any.
    ///
    /// An inverted adaptive range has no valid implicit current value. This
    /// fails closed as `None` instead of silently selecting one invalid bound.
    pub fn current(&self) -> Option<T> {
        match self.contract {
            ElasticValue::Fixed(value) | ElasticValue::Pinned(value) => Some(value),
            ElasticValue::Adaptive { min, max } if min <= max => {
                self.assigned.or(Some(max))
            }
            ElasticValue::Adaptive { .. } => None,
            ElasticValue::Auto => self.assigned,
        }
    }

    /// Try to assign a new value; rejects values outside the contract.
    pub fn assign(&mut self, value: T) -> Result<(), &'static str> {
        if self.contract.allows(value) {
            self.assigned = Some(value);
            Ok(())
        } else {
            Err("value outside the elastic contract")
        }
    }

    /// Whether the controller may change the value.
    pub fn is_adaptive(&self) -> bool {
        self.contract.is_adaptive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_cannot_change() {
        let mut value = AdaptiveValue::new(ElasticValue::Fixed(16384u64));
        assert_eq!(value.current(), Some(16384));
        assert!(value.assign(32768).is_err());
        assert!(!value.is_adaptive());
    }

    #[test]
    fn auto_owned_by_controller() {
        let mut value = AdaptiveValue::new(ElasticValue::Auto);
        assert_eq!(value.current(), None);
        assert!(value.is_adaptive());
        value.assign(4096).unwrap();
        assert_eq!(value.current(), Some(4096));
    }

    #[test]
    fn adaptive_range_enforced() {
        let mut value = AdaptiveValue::new(ElasticValue::Adaptive {
            min: 1024,
            max: 65536,
        });
        assert_eq!(value.current(), Some(65536));
        assert!(value.is_adaptive());
        assert!(value.assign(512).is_err());
        assert!(value.assign(65536).is_ok());
        assert!(value.assign(70000).is_err());
    }

    #[test]
    fn inverted_adaptive_range_fails_closed() {
        let mut value = AdaptiveValue::new(ElasticValue::Adaptive {
            min: 65536u64,
            max: 1024u64,
        });
        assert_eq!(value.current(), None);
        assert!(value.assign(4096).is_err());
    }

    #[test]
    fn pinned_protected() {
        let mut value = AdaptiveValue::new(ElasticValue::Pinned(128u64));
        assert_eq!(value.current(), Some(128));
        assert!(value.assign(64).is_err());
        assert!(!value.is_adaptive());
    }
}
