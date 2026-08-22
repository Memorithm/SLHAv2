//! elastic-runtime — std-dependent runtime for the Elastic resource language.
//!
//! Provides:
//! - [`telemetry`]: structured resource telemetry (bytes, pressure, counters).
//! - [`journal`]: a compact deterministic decision/adaptation journal.
//! - [`coordinator`]: hierarchical multi-resource coordination.
//!
//! The runtime depends only on `elastic-core` — never on SLHAv2 or any
//! application crate. Deterministic mode is the default contract: journals
//! use logical steps, and telemetry is a plain struct (no wall-clock coupling).

pub mod coordinator;
pub mod journal;
pub mod telemetry;
