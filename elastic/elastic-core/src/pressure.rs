//! Pressure: normalized resource pressure levels and smoothing.
//!
//! Pressure is the primary controller input. It is normalized to a
//! `0.0..=1.0` scale where higher means more constrained. Levels are derived
//! from watermarks, never from magic constants in controller code.

/// Coarse pressure exposure levels.
///
/// Critical hard conditions MUST override weighted averages: a controller
/// that reports `Low` because latency is excellent while an OOM is imminent
/// is broken. Critical is a hard condition, not a soft level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureLevel {
    /// Resource is comfortably available.
    Low,
    /// Resource is in the normal operating band.
    Normal,
    /// Resource is constrained; adaptation should be considered.
    High,
    /// Hard condition: imminent exhaustion. Overrides any utility calculus.
    Critical,
}

impl PressureLevel {
    /// Whether this level demands immediate action regardless of utility.
    pub fn is_critical(&self) -> bool {
        matches!(self, PressureLevel::Critical)
    }

    /// Whether this level permits (or demands) adaptation.
    pub fn demands_action(&self) -> bool {
        matches!(self, PressureLevel::High | PressureLevel::Critical)
    }
}

/// A measured pressure sample: normalized value plus the level derived from
/// the configured watermarks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pressure {
    /// Normalized pressure in `[0.0, 1.0]` (1.0 = exhausted).
    pub value: f64,
    /// Coarse level derived from the watermarks.
    pub level: PressureLevel,
}

impl Pressure {
    /// Compute the normalized pressure and level for `used` of `capacity`
    /// with the given hysteresis watermarks.
    ///
    /// - `used > capacity` → `Critical` at 1.0 (hard overflow).
    /// - `value >= high` → `High`.
    /// - `value <= low` → `Low`.
    /// - otherwise → `Normal`.
    pub fn from_used(used: u64, capacity: u64, wm: Watermarks) -> Self {
        if capacity == 0 {
            return Self {
                value: if used == 0 { 0.0 } else { 1.0 },
                level: if used == 0 {
                    PressureLevel::Low
                } else {
                    PressureLevel::Critical
                },
            };
        }
        let value = (used as f64 / capacity as f64).clamp(0.0, 1.0);
        let level = if used > capacity {
            PressureLevel::Critical
        } else if value >= wm.high {
            PressureLevel::High
        } else if value <= wm.low {
            PressureLevel::Low
        } else {
            PressureLevel::Normal
        };
        Self { value, level }
    }
}

/// High/low watermarks for hysteresis (see [`crate::hysteresis`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Watermarks {
    /// Crossing this (inclusive) raises pressure to `High`.
    pub high: f64,
    /// Falling to this (inclusive) restores `Low`.
    pub low: f64,
}

impl Watermarks {
    /// Standard hysteresis pair: HIGH = 0.85, LOW = 0.70.
    pub const fn standard() -> Self {
        Self {
            high: 0.85,
            low: 0.70,
        }
    }

    /// Create watermarks, asserting `0 <= low < high <= 1`.
    pub const fn new(low: f64, high: f64) -> Self {
        assert!(low >= 0.0 && high <= 1.0 && low < high);
        Self { high, low }
    }
}

/// Exponential moving average for pressure smoothing.
///
/// Deterministic: the same samples in the same order produce the same output.
/// `alpha` in `(0, 1)`; larger = faster response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureSmoother {
    alpha: f64,
    current: Option<f64>,
}

impl PressureSmoother {
    /// Create a smoother with the given `alpha`.
    pub const fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0);
        Self {
            alpha,
            current: None,
        }
    }

    /// Push a raw sample; returns the smoothed value.
    pub fn push(&mut self, raw: f64) -> f64 {
        self.current = Some(match self.current {
            None => raw,
            Some(prev) => self.alpha * raw + (1.0 - self.alpha) * prev,
        });
        self.current.expect("just set")
    }

    /// The current smoothed value, if any.
    pub fn value(&self) -> Option<f64> {
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_levels_follow_watermarks() {
        let wm = Watermarks::standard();
        assert_eq!(Pressure::from_used(50, 100, wm).level, PressureLevel::Low);
        assert_eq!(
            Pressure::from_used(75, 100, wm).level,
            PressureLevel::Normal
        );
        assert_eq!(Pressure::from_used(85, 100, wm).level, PressureLevel::High);
        assert_eq!(Pressure::from_used(90, 100, wm).level, PressureLevel::High);
    }

    #[test]
    fn overflow_is_critical() {
        let wm = Watermarks::standard();
        let p = Pressure::from_used(101, 100, wm);
        assert_eq!(p.level, PressureLevel::Critical);
        assert_eq!(p.value, 1.0);
    }

    #[test]
    fn zero_capacity() {
        let wm = Watermarks::standard();
        assert_eq!(Pressure::from_used(0, 0, wm).level, PressureLevel::Low);
        assert_eq!(Pressure::from_used(1, 0, wm).level, PressureLevel::Critical);
    }

    #[test]
    fn smoother_is_deterministic_ewma() {
        let mut a = PressureSmoother::new(0.5);
        let mut b = PressureSmoother::new(0.5);
        for v in [0.1, 0.9, 0.2, 0.8] {
            assert_eq!(a.push(v), b.push(v));
        }
        // EWMA with alpha=0.5: 0.1 -> 0.5 -> 0.35 -> 0.575
        assert!((a.value().unwrap() - 0.575).abs() < 1e-12);
    }
}
