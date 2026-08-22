//! Compact deterministic decision/adaptation journal.
//!
//! Suitable for deterministic replay, audit, debugging and research. Each
//! entry records the observation id, resource id, previous state, pressure,
//! forecast, action, reason, costs, resulting state, verification and logical
//! step. Deterministic mode prefers logical steps over wall-clock time.

use elastic_core::decision::DecisionTrace;
use elastic_core::LogicalStep;

/// One journal entry (an adaptation or a no-op decision).
#[derive(Clone, Debug, PartialEq)]
pub struct JournalEntry {
    /// Logical step (injected time in deterministic mode).
    pub step: LogicalStep,
    /// Resource identifier.
    pub resource_id: String,
    /// State id before the decision.
    pub state_id: u64,
    /// Pressure before the decision.
    pub pressure: f64,
    /// Forecast pressure.
    pub forecast: Option<f64>,
    /// Chosen action (or "none").
    pub action: String,
    /// Stable reason code.
    pub reason_code: Option<String>,
    /// Expected cost.
    pub expected_cost: f64,
    /// Actual outcome (filled after execution).
    pub actual_outcome: Option<String>,
    /// Verification result.
    pub verified: Option<bool>,
    /// Resulting state id (filled after execution).
    pub resulting_state: Option<u64>,
}

/// An append-only journal.
#[derive(Clone, Debug, Default)]
pub struct ElasticJournal {
    entries: Vec<JournalEntry>,
    /// Max entries kept (bounded memory); 0 = unbounded.
    max_entries: usize,
}

impl ElasticJournal {
    /// Create an unbounded journal.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a journal bounded to `max_entries` (oldest dropped).
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Append a decision trace and its outcome.
    pub fn push(
        &mut self,
        resource_id: &str,
        trace: &DecisionTrace,
        actual_outcome: Option<&str>,
        verified: Option<bool>,
        resulting_state: Option<u64>,
    ) {
        self.entries.push(JournalEntry {
            step: trace.step,
            resource_id: resource_id.to_string(),
            state_id: trace.state_id,
            pressure: trace.measured_pressure,
            forecast: trace.forecast_pressure,
            action: trace.chosen.unwrap_or("none").to_string(),
            reason_code: trace.reason.map(|r| r.code().to_string()),
            expected_cost: trace.expected_cost,
            actual_outcome: actual_outcome.map(str::to_string),
            verified,
            resulting_state,
        });
        if self.max_entries > 0 && self.entries.len() > self.max_entries {
            let excess = self.entries.len() - self.max_entries;
            self.entries.drain(0..excess);
        }
    }

    /// All entries in order.
    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the journal is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Replay the action sequence: the list of chosen actions in order.
    pub fn replay_actions(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.action.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use elastic_core::decision::DecisionTrace;
    use elastic_core::pressure::PressureLevel;
    use elastic_core::reason::Reason;

    fn trace(step: u64, action: &'static str) -> DecisionTrace {
        let mut t = DecisionTrace::new(step, step * 7);
        t.measured_pressure = 0.9;
        t.pressure_level = PressureLevel::High;
        t.forecast_pressure = Some(0.95);
        t.choose(action, Reason::new("pressure_high"), 10.0, 1.0);
        t
    }

    #[test]
    fn journal_roundtrip() {
        let mut j = ElasticJournal::new();
        j.push(
            "ctx",
            &trace(1, "demote"),
            Some("demoted"),
            Some(true),
            Some(99),
        );
        j.push("ctx", &trace(2, "none"), None, Some(true), Some(99));
        assert_eq!(j.len(), 2);
        assert_eq!(j.replay_actions(), vec!["demote", "none"]);
        assert_eq!(j.entries()[0].reason_code.as_deref(), Some("pressure_high"));
        assert_eq!(j.entries()[0].expected_cost, 1.0);
    }

    #[test]
    fn journal_bounded() {
        let mut j = ElasticJournal::with_capacity(3);
        for s in 0..10 {
            j.push("ctx", &trace(s, "none"), None, None, None);
        }
        assert_eq!(j.len(), 3);
        assert_eq!(j.entries()[0].step, 7);
    }
}
