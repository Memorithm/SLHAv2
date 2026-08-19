//! Hysteresis and anti-thrash: watermarks, cooldowns and minimum useful
//! adaptation.
//!
//! Elastic does NOT mean constant oscillation. The controller must not
//! alternate compress/decompress around one noisy threshold. This module
//! provides the explicit machinery: high/low watermarks, a cooldown window
//! after an adaptation, and a minimum-adaptation gate.

/// A hysteresis gate for a single binary decision (e.g. "demote").
///
/// State is `Idle | Active`, where `Active` means the action has been taken
/// and will be kept until the low watermark is restored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HysteresisState {
    /// No action currently taken.
    Idle,
    /// Action taken; will be released only below the low watermark.
    Active,
}

/// Anti-thrash gate combining watermarks, cooldown and a minimum
/// adaptation interval.
#[derive(Clone, Copy, Debug)]
pub struct HysteresisGate {
    /// High watermark (enter).
    pub high: f64,
    /// Low watermark (exit).
    pub low: f64,
    /// Steps to wait after entering before any further *opposite* action.
    pub cooldown_steps: u64,
    /// Minimum number of steps between two adaptations in the same direction.
    pub min_interval_steps: u64,
    state: HysteresisState,
    last_action_step: u64,
}

impl HysteresisGate {
    /// Create a gate with watermarks and anti-thrash parameters.
    pub const fn new(high: f64, low: f64, cooldown_steps: u64, min_interval_steps: u64) -> Self {
        assert!(low < high);
        Self {
            high,
            low,
            cooldown_steps,
            min_interval_steps,
            state: HysteresisState::Idle,
            last_action_step: u64::MAX,
        }
    }

    /// Standard gate: HIGH=0.85, LOW=0.70, cooldown 3 steps.
    pub const fn standard() -> Self {
        Self::new(0.85, 0.70, 3, 2)
    }

    /// Step at which the gate last took an action (`u64::MAX` = never).
    pub fn last_action_step(&self) -> u64 {
        self.last_action_step
    }

    /// Feed a smoothed pressure sample at `step`; returns whether the action
    /// should be taken or kept.
    ///
    /// - `Active` stays active until `value <= low`.
    /// - `Idle` becomes active when `value >= high` AND the minimum-interval
    ///   constraint passes. The very first activation is always allowed.
    pub fn update(&mut self, value: f64, step: u64) -> bool {
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
                let interval_ok = self.last_action_step == u64::MAX
                    || step.saturating_sub(self.last_action_step) >= self.min_interval_steps;
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

    /// The cooldown check for the *opposite* action (e.g. promote right after
    /// a demote): refuses for `cooldown_steps` after the last action.
    pub fn in_cooldown(&self, step: u64) -> bool {
        if self.last_action_step == u64::MAX {
            return false;
        }
        step.saturating_sub(self.last_action_step) < self.cooldown_steps
    }

    /// Whether the minimum interval between two actions has elapsed.
    pub fn in_min_interval(&self, step: u64) -> bool {
        if self.last_action_step == u64::MAX {
            return false;
        }
        step.saturating_sub(self.last_action_step) < self.min_interval_steps
    }

    /// Record an opposite-direction action (e.g. a promote after a demote)
    /// so the cooldown applies to it as well.
    pub fn note_opposite_action(&mut self, step: u64) {
        self.last_action_step = step;
    }

    /// Current state.
    pub fn state(&self) -> HysteresisState {
        self.state
    }
}

impl HysteresisState {
    /// Whether the action is currently active (taken and held).
    pub fn is_active(&self) -> bool {
        matches!(self, HysteresisState::Active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing_high_enters_crossing_low_exits() {
        let mut g = HysteresisGate::standard();
        assert!(!g.update(0.5, 1));
        assert!(!g.update(0.8, 2));
        assert!(g.update(0.86, 3)); // >= 0.85
        assert!(g.update(0.84, 4)); // stays active even below high
        assert!(g.update(0.75, 5)); // still above low
        assert!(!g.update(0.69, 6)); // <= 0.70 -> exit
    }

    #[test]
    fn min_interval_blocks_rapid_reentry() {
        let mut g = HysteresisGate::new(0.8, 0.5, 0, 5);
        assert!(g.update(0.9, 1));
        // Exit below low first, then try to re-enter quickly.
        assert!(!g.update(0.4, 2));
        assert!(!g.update(0.9, 2)); // within min interval (1 -> 2)
        assert!(!g.update(0.9, 5)); // 5-1 = 4 < 5
        assert!(g.update(0.9, 7)); // 7-1 = 6 >= 5 -> allowed
    }

    #[test]
    fn cooldown_blocks_opposite_action() {
        let mut g = HysteresisGate::new(0.85, 0.70, 3, 2);
        assert!(g.update(0.9, 1));
        assert!(g.in_cooldown(2));
        assert!(!g.in_cooldown(10));
    }

    #[test]
    fn no_oscillation_around_threshold() {
        // Pressure oscillating around 0.80 must not flip the gate.
        let mut g = HysteresisGate::standard();
        let mut actions = 0;
        for step in 1..=100 {
            let v = if step % 2 == 0 { 0.81 } else { 0.79 };
            if g.update(v, step) {
                actions += 1;
            }
        }
        assert!(actions <= 1, "oscillated {actions} times");
    }
}
