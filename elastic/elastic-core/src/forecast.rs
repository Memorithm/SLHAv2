//! Deterministic forecasting: EWMA level, bounded trend and linear projection.
//!
//! The baseline controller does not require machine learning. Non-finite
//! samples are rejected by preserving the previous finite forecast state.

fn step_scaled(trend: f64, steps: u64) -> f64 {
    trend * steps as f64
}

/// Deterministic EWMA level + bounded EWMA trend forecast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forecast {
    level: f64,
    trend: f64,
    max_trend: f64,
    alpha: f64,
    beta: f64,
    initialized: bool,
}

impl Forecast {
    /// Create a forecast with smoothing constants in `(0,1)` and a finite,
    /// non-negative trend bound.
    pub const fn new(alpha: f64, beta: f64, max_trend: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0);
        assert!(beta > 0.0 && beta < 1.0);
        assert!(max_trend >= 0.0 && max_trend < f64::INFINITY);
        Self {
            level: 0.0,
            trend: 0.0,
            max_trend,
            alpha,
            beta,
            initialized: false,
        }
    }

    /// Push one observation and return the one-step forecast.
    ///
    /// `NaN`/infinite input is ignored so it cannot poison all future control
    /// decisions. The previous forecast (or 0 before initialization) is
    /// returned unchanged.
    pub fn observe(&mut self, value: f64) -> f64 {
        if !value.is_finite() {
            return self.forecast(1);
        }
        if !self.initialized {
            self.level = value;
            self.trend = 0.0;
            self.initialized = true;
            return value;
        }
        let prediction = self.level + self.trend;
        let error = value - prediction;
        self.level += self.alpha * error;
        self.trend += self.beta * error;
        self.trend = self.trend.clamp(-self.max_trend, self.max_trend);
        self.level + self.trend
    }

    /// Forecast `steps` steps ahead without mutating state.
    pub fn forecast(&self, steps: u64) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.level + step_scaled(self.trend, steps)
    }

    /// Whether at least one finite observation has been accepted.
    pub fn initialized(&self) -> bool {
        self.initialized
    }
}

/// Time until a growing finite series crosses `capacity`, in logical steps.
///
/// Returns `Some(0)` when already exhausted and `None` when the series is
/// uninitialized, flat/falling, or any required value is non-finite/invalid.
pub fn steps_to_exhaustion(forecast: &Forecast, capacity: f64) -> Option<u64> {
    if !forecast.initialized() || !capacity.is_finite() || capacity < 0.0 {
        return None;
    }
    let level = forecast.level;
    let trend = forecast.trend;
    if !level.is_finite() || !trend.is_finite() {
        return None;
    }
    if level >= capacity {
        return Some(0);
    }
    if trend <= 0.0 {
        return None;
    }
    let raw = (capacity - level) / trend;
    if !raw.is_finite() || raw < 0.0 || raw > u64::MAX as f64 {
        return None;
    }
    let truncated = raw as u64;
    Some(if truncated as f64 == raw {
        truncated
    } else {
        truncated.saturating_add(1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_converges_on_constant_series() {
        let mut forecast = Forecast::new(0.5, 0.1, 1.0);
        for _ in 0..20 {
            forecast.observe(100.0);
        }
        assert!((forecast.forecast(1) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn forecast_tracks_linear_growth() {
        let mut forecast = Forecast::new(0.3, 0.3, 10.0);
        let mut value = 0.0;
        for _ in 0..50 {
            value += 1.0;
            forecast.observe(value);
        }
        let ahead = forecast.forecast(10);
        assert!(ahead > value, "ahead={ahead} last={value}");
        assert!(ahead < 200.0, "ahead={ahead}");
    }

    #[test]
    fn bounded_derivative_prevents_spike_explosion() {
        let mut forecast = Forecast::new(0.5, 0.5, 1.0);
        forecast.observe(0.0);
        forecast.observe(0.0);
        forecast.observe(1000.0);
        assert!(forecast.forecast(1_000_000) < 2_000_000.0);
    }

    #[test]
    fn exhaustion_timing() {
        let mut forecast = Forecast::new(0.5, 0.4, 10.0);
        let mut value = 90.0;
        for _ in 0..30 {
            value += 1.0;
            forecast.observe(value);
        }
        let steps = steps_to_exhaustion(&forecast, 200.0).unwrap();
        assert!((1..=200).contains(&steps));
    }

    #[test]
    fn exhaustion_none_when_flat_or_invalid() {
        let mut forecast = Forecast::new(0.5, 0.1, 1.0);
        forecast.observe(50.0);
        forecast.observe(50.0);
        assert_eq!(steps_to_exhaustion(&forecast, 100.0), None);
        assert_eq!(steps_to_exhaustion(&forecast, f64::NAN), None);
        assert_eq!(steps_to_exhaustion(&forecast, -1.0), None);
    }

    #[test]
    fn non_finite_sample_does_not_poison_history() {
        let mut forecast = Forecast::new(0.5, 0.2, 1.0);
        forecast.observe(10.0);
        let before = forecast.forecast(3);
        assert_eq!(forecast.observe(f64::NAN), forecast.forecast(1));
        assert_eq!(forecast.forecast(3), before);
        assert!(forecast.forecast(3).is_finite());
    }

    #[test]
    fn deterministic_same_history_same_forecast() {
        let mut a = Forecast::new(0.4, 0.2, 5.0);
        let mut b = Forecast::new(0.4, 0.2, 5.0);
        for value in [1.0, 2.0, 3.5, 5.0, 8.0] {
            a.observe(value);
            b.observe(value);
        }
        assert_eq!(a.forecast(3), b.forecast(3));
    }
}
