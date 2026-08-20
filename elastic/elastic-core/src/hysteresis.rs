//! Hysteresis and anti-thrash: watermarks, cooldowns and minimum useful
//! adaptation.
//!
//! Elastic does NOT mean constant oscillation. The controller must not
//! alternate compress/decompress around one noisy threshold. This module
//! provides the explicit machinery: high/low watermarks, a cooldown window
//! after an adaptation, and a minimum interval between repeated adaptations.

/// A tier of a binary pressure gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HysteresisState {
    /// No pressure action currently held.
    Idle,
    /// Pressure action is active; it remains active until the low watermark.
    Active,
}

/// Anti-thrash gate combining watermarks, cooldown and minimum action interval.
#[derive(Clone, Copy, Debug)]
pub struct HysteresisGate {
    /// High watermark (enter).
    pub high: f64,
    /// Low watermark (exit).
    pub low: f64,
    /// Steps to wait before an opposite-direction action.
    pub cooldown_steps: u64,
    /// Minimum steps between two adaptation actions.
    pub min_interval_steps: u64,
    state: HysteresisState,
    last_action_step: u64,
}

impl HysteresisGate {
    /// Create a gate with validated watermarks and anti-thrash parameters.
    pub const fn new(high: f64, low: f64, cooldown_steps: u64, min_interval_steps: u64) -> Self {
        assert!(low >= 0.0 && high <= 1.0 && low < high);
        Self {
            high,
            low,
            cooldown_steps,
            min_interval_steps,
            state: HysteresisState::Idle,
            last_action_step: u64::MAX,
        }
    }

    /// Standard gate: HIGH=0.85, LOW=0.70, cooldown 3, min interval 2.
    pub const fn standard() -> Self {
        Self::new(0.85, 0.70, 3, 2)
    }

    /// Step of the last real adaptation (`u64::MAX` = never).
    pub fn last_action_step(&self) -> u64 {
        self.last_action_step
    }

    /// Feed a smoothed pressure sample.
    ///
    /// This method updates only the pressure state. Entering `Active` records
    /// the triggering step because the caller is expected to act immediately;
    /// later repeated actions must call [`Self::note_action`] explicitly.
    pub fn update(&mut self, value: f64, step: u64) -> bool {
        if !value.is_finite() {
            return self.state == HysteresisState::Active;
        }
        match self.state {
            HysteresisState::Active => {
                if value <= self.low {
                    self.state = HysteresisState::Idle;
                    false
                } else {
                    true
                }
            }
            HysteresisState::Idle => {
                let interval_ok = !self.in_min_interval(step);
                if value >= self.high && interval_ok {
                    self.state = HysteresisState::Active;
                    self.last_action_step = step;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Whether an opposite-direction action is still in cooldown.
    pub fn in_cooldown(&self, step: u64) -> bool {
        if self.last_action_step == u64::MAX {
            return false;
        }
        step.saturating_sub(self.last_action_step) < self.cooldown_steps
    }

    /// Whether another adaptation would violate the minimum action interval.
    pub fn in_min_interval(&self, step: u64) -> bool {
        if self.last_action_step == u64::MAX {
            return false;
        }
        step.saturating_sub(self.last_action_step) < self.min_interval_steps
    }

    /// Record any real adaptation (same or opposite direction).
    pub fn note_action(&mut self, step: u64) {
        self.last_action_step = step;
    }

    /// Compatibility alias for older callers.
    pub fn note_opposite_action(&mut self, step: u64) {
        self.note_action(step);
    }

    /// Current pressure state.
    pub fn state(&self) -> HysteresisState {
        self.state
    }
}

impl HysteresisState {
    /// Whether the pressure action is currently held.
    pub fn is_active(&self) -> bool {
        matches!(self, HysteresisState::Active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing_high_enters_crossing_low_exits() {
        let mut gate = HysteresisGate::standard();
        assert!(!gate.update(0.5, 1));
        assert!(!gate.update(0.8, 2));
        assert!(gate.update(0.86, 3));
        assert!(gate.update(0.84, 4));
        assert!(gate.update(0.75, 5));
        assert!(!gate.update(0.69, 6));
    }

    #[test]
    fn min_interval_blocks_rapid_reentry() {
        let mut gate = HysteresisGate::new(0.8, 0.5, 0, 5);
        assert!(gate.update(0.9, 1));
        assert!(!gate.update(0.4, 2));
        assert!(!gate.update(0.9, 2));
        assert!(!gate.update(0.9, 5));
        assert!(gate.update(0.9, 7));
    }

    #[test]
    fn repeated_action_rearms_min_interval() {
        let mut gate = HysteresisGate::new(0.8, 0.5, 3, 2);
        assert!(gate.update(0.9, 1));
        assert!(gate.in_min_interval(2));
        assert!(!gate.in_min_interval(3));
        gate.note_action(3);
        assert!(gate.in_min_interval(4));
        assert!(!gate.in_min_interval(5));
        assert!(gate.in_cooldown(4));
        assert!(!gate.in_cooldown(6));
    }

    #[test]
    fn non_finite_sample_does_not_change_state() {
        let mut gate = HysteresisGate::standard();
        assert!(!gate.update(f64::NAN, 1));
        assert_eq!(gate.state(), HysteresisState::Idle);
        assert!(gate.update(0.9, 2));
        assert!(gate.update(f64::NAN, 3));
        assert_eq!(gate.state(), HysteresisState::Active);
    }

    #[test]
    fn no_activation_around_mid_band() {
        let mut gate = HysteresisGate::standard();
        for step in 1..=100 {
            let value = if step % 2 == 0 { 0.81 } else { 0.79 };
            assert!(!gate.update(value, step));
        }
    }
}
