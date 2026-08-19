//! Deterministic forecasting: EWMA, moving trend and bounded linear forecast.
//!
//! The baseline controller does not require machine learning. These
//! forecasters are deterministic functions of their input history; learned
//! policies belong in an experimental layer, never here.

/// Add a u64 step count to an f64 trend without pulling in `std`
/// (`f64 + u64` is std-only; we scale explicitly).
fn step_scaled(trend: f64, steps: u64) -> f64 {
    trend * steps as f64
}

/// A deterministic linear forecast of a byte/resource series.
///
/// Uses EWMA level + EWMA trend with bounded derivative, so a single spike
/// cannot produce an absurd forecast. `alpha` and `beta` in `(0,1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forecast {
    /// EWMA level.
    level: f64,
    /// EWMA trend (per step).
    trend: f64,
    /// Maximum absolute trend per step (bounds the derivative).
    max_trend: f64,
    alpha: f64,
    beta: f64,
    initialized: bool,
}

impl Forecast {
    /// Create a forecast with the given smoothing constants and trend bound.
    pub const fn new(alpha: f64, beta: f64, max_trend: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0);
        assert!(beta > 0.0 && beta < 1.0);
        assert!(max_trend >= 0.0);
        Self {
            level: 0.0,
            trend: 0.0,
            max_trend,
            alpha,
            beta,
            initialized: false,
        }
    }

    /// Push an observation; returns the one-step forecast for the next step.
    pub fn observe(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.level = value;
            self.trend = 0.0;
            self.initialized = true;
            return value;
        }
        let pred = self.level + self.trend;
        let error = value - pred;
        self.level += self.alpha * error;
        self.trend += self.beta * error;
        self.trend = self.trend.clamp(-self.max_trend, self.max_trend);
        self.level + self.trend
    }

    /// Forecast `steps` steps ahead from the current state (no new
    /// observations). Deterministic.
    pub fn forecast(&self, steps: u64) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.level + step_scaled(self.trend, steps)
    }

    /// Whether any observation has been seen.
    pub fn initialized(&self) -> bool {
        self.initialized
    }
}

/// The time until a growing series crosses `capacity`, in steps.
///
/// Returns `None` when the trend is not positive or the series is already
/// past capacity (the caller should treat that as immediate exhaustion).
pub fn steps_to_exhaustion(forecast: &Forecast, capacity: f64) -> Option<u64> {
    if !forecast.initialized() {
        return None;
    }
    let level = forecast.level;
    let trend = forecast.trend;
    if level >= capacity {
        return Some(0);
    }
    if trend <= 0.0 {
        return None;
    }
    // ceil without std: truncate-toward-zero then adjust for positives.
    let raw = (capacity - level) / trend;
    let steps = if raw < 0.0 {
        0.0
    } else {
        let t = raw as u64 as f64;
        if t == raw {
            t
        } else {
            t + 1.0
        }
    };
    if steps.is_finite() {
        Some(steps as u64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_converges_on_constant_series() {
        let mut f = Forecast::new(0.5, 0.1, 1.0);
        for _ in 0..20 {
            f.observe(100.0);
        }
        assert!((f.forecast(1) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn forecast_tracks_linear_growth() {
        let mut f = Forecast::new(0.3, 0.3, 10.0);
        let mut v = 0.0;
        for _ in 0..50 {
            v += 1.0;
            f.observe(v);
        }
        // Final observed value is 50; the trend is positive, so the 10-step
        // forecast must exceed the last observation (and stay bounded).
        let ahead = f.forecast(10);
        assert!(ahead > v, "ahead={ahead} last={v}");
        assert!(ahead < 200.0, "ahead={ahead}");
    }

    #[test]
    fn bounded_derivative_prevents_spike_explosion() {
        let mut f = Forecast::new(0.5, 0.5, 1.0);
        f.observe(0.0);
        f.observe(0.0);
        f.observe(1000.0); // spike
                           // Even with beta=0.5, the trend is clamped to 1.0/step, so the
                           // long-range forecast cannot blow up.
        assert!(f.forecast(1_000_000) < 2_000_000.0);
    }

    #[test]
    fn exhaustion_timing() {
        // Build an explicitly increasing series so the trend is positive.
        let mut f = Forecast::new(0.5, 0.4, 10.0);
        let mut v = 90.0;
        for _ in 0..30 {
            v += 1.0;
            f.observe(v);
        }
        assert!(f.trend > 0.0, "trend={}", f.trend);
        let steps = steps_to_exhaustion(&f, 200.0);
        assert!(steps.is_some(), "trend={} level={}", f.trend, f.level);
        assert!((1..=200).contains(&steps.unwrap()), "steps={steps:?}");
    }

    #[test]
    fn exhaustion_none_when_flat() {
        let mut f = Forecast::new(0.5, 0.1, 1.0);
        f.observe(50.0);
        f.observe(50.0);
        assert_eq!(steps_to_exhaustion(&f, 100.0), None);
    }

    #[test]
    fn deterministic_same_history_same_forecast() {
        let mut a = Forecast::new(0.4, 0.2, 5.0);
        let mut b = Forecast::new(0.4, 0.2, 5.0);
        for v in [1.0, 2.0, 3.5, 5.0, 8.0] {
            a.observe(v);
            b.observe(v);
        }
        assert_eq!(a.forecast(3), b.forecast(3));
    }
}
