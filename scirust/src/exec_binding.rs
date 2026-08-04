//! Strict, structural parsing of the frozen A2 execution binding.
//!
//! The execution binding ties the frozen A2 experiment definition to the
//! concrete immutable inputs a production run actually consumes: the
//! normalisation manifest, the v1-to-v2 equivalence proof, the strict-parser
//! audit, the dataset, the splits, and the exact parser/trainer commit and
//! binary. It changes no objective, hyperparameter, dataset, split or variant
//! definition.
//!
//! Parsing is typed and structural for the same reason the manifest parser is:
//! a substring loader silently accepted a nested same-name field, trailing data
//! and a duplicated key. Every rejection carries a stable machine-readable code
//! and must happen before any staging directory exists.

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingError {
    Missing(String),
    HashInvalid(String),
    HashMismatch {
        expected: String,
        actual: String,
    },
    JsonInvalid(String),
    TrailingData(String),
    DuplicateField(String),
    UnknownField(String),
    SchemaUnsupported(String),
    Invalid(String),
    Mismatch {
        field: String,
        expected: String,
        actual: String,
    },
}

impl BindingError {
    /// Stable code for a mismatching field. Audits assert these, not exit codes.
    pub fn code(&self) -> &'static str {
        match self {
            BindingError::Missing(_) => "EXEC_BINDING_MISSING",
            BindingError::HashInvalid(_) => "EXEC_BINDING_HASH_INVALID",
            BindingError::HashMismatch { .. } => "EXEC_BINDING_HASH_MISMATCH",
            BindingError::JsonInvalid(_) => "EXEC_BINDING_JSON_INVALID",
            BindingError::TrailingData(_) => "EXEC_BINDING_TRAILING_DATA",
            BindingError::DuplicateField(_) => "EXEC_BINDING_DUPLICATE_FIELD",
            BindingError::UnknownField(_) => "EXEC_BINDING_UNKNOWN_FIELD",
            BindingError::SchemaUnsupported(_) => "EXEC_BINDING_SCHEMA_UNSUPPORTED",
            BindingError::Invalid(_) => "EXEC_BINDING_INVALID",
            BindingError::Mismatch { field, .. } => match field.as_str() {
                "a2_sha256" => "EXEC_BINDING_A2_MISMATCH",
                "scope_clarification_sha256" => "EXEC_BINDING_SCOPE_MISMATCH",
                "manifest_v2_sha256" => "EXEC_BINDING_MANIFEST_MISMATCH",
                "v1_v2_equivalence_report_sha256" => "EXEC_BINDING_EQUIVALENCE_MISMATCH",
                "dataset_aggregate_sha256" => "EXEC_BINDING_DATASET_MISMATCH",
                "row_reconciliation_sha256" => "EXEC_BINDING_RECONCILIATION_MISMATCH",
                "training_split_sha256" | "split_manifest_sha256" => "EXEC_BINDING_SPLIT_MISMATCH",
                "strict_parser_commit" => "EXEC_BINDING_PARSER_COMMIT_MISMATCH",
                "trainer_commit" => "EXEC_BINDING_TRAINER_COMMIT_MISMATCH",
                "trainer_binary_sha256" => "EXEC_BINDING_TRAINER_BINARY_MISMATCH",
                _ => "EXEC_BINDING_INVALID",
            },
        }
    }
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:?}", self.code(), self)
    }
}

pub const EXPECTED_SCHEMA: &str = "slha_a2_execution_binding_v2";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionBinding {
    pub schema: String,
    pub statement: String,
    pub a2_sha256: String,
    pub scope_clarification_sha256: String,
    pub manifest_v1_sha256: String,
    pub manifest_v2_sha256: String,
    pub v1_v2_equivalence_report_sha256: String,
    pub strict_manifest_parser_audit_sha256: String,
    pub dataset_aggregate_sha256: String,
    pub row_reconciliation_sha256: String,
    pub split_manifest_sha256: String,
    pub training_split_sha256: String,
    pub ranking_validation_split_sha256: String,
    pub training_diagnostics_split_sha256: String,
    pub strict_parser_commit: String,
    pub trainer_commit: String,
    pub trainer_binary_sha256: String,
    pub valid: bool,
}

/// What the caller requires the binding to declare.
pub struct Expected<'a> {
    pub a2: &'a str,
    pub scope_clarification: &'a str,
    pub manifest_v2: &'a str,
    pub equivalence: &'a str,
    pub dataset: &'a str,
    pub reconciliation: &'a str,
    pub training_split: &'a str,
    pub strict_parser_commit: &'a str,
    pub trainer_commit: &'a str,
    pub trainer_binary: &'a str,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn want(field: &str, actual: &str, expected: &str) -> Result<(), BindingError> {
    if actual == expected {
        Ok(())
    } else {
        Err(BindingError::Mismatch {
            field: field.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

/// Parse and fully verify the execution binding.
///
/// `actual_file_sha256` is the hash of `bytes` as computed by the caller;
/// `expected_file_sha256` is what the operator asserted on the command line.
pub fn parse_and_verify(
    bytes: &[u8],
    actual_file_sha256: &str,
    expected_file_sha256: &str,
    exp: &Expected<'_>,
) -> Result<ExecutionBinding, BindingError> {
    if !is_sha256(expected_file_sha256) {
        return Err(BindingError::HashInvalid(expected_file_sha256.to_string()));
    }
    if actual_file_sha256 != expected_file_sha256 {
        return Err(BindingError::HashMismatch {
            expected: expected_file_sha256.to_string(),
            actual: actual_file_sha256.to_string(),
        });
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|e| BindingError::JsonInvalid(format!("not UTF-8: {e}")))?;

    let mut de = serde_json::Deserializer::from_str(text);
    let b = ExecutionBinding::deserialize(&mut de).map_err(|e| {
        let s = e.to_string();
        if s.contains("duplicate field") {
            BindingError::DuplicateField(s)
        } else if s.contains("unknown field") {
            BindingError::UnknownField(s)
        } else {
            BindingError::JsonInvalid(s)
        }
    })?;
    // rejects data after the closing brace; a plain from_str would not
    de.end()
        .map_err(|e| BindingError::TrailingData(e.to_string()))?;

    if b.schema != EXPECTED_SCHEMA {
        return Err(BindingError::SchemaUnsupported(b.schema));
    }
    if !b.valid {
        return Err(BindingError::Invalid("valid != true".into()));
    }
    for (name, v) in [
        ("a2_sha256", &b.a2_sha256),
        ("manifest_v2_sha256", &b.manifest_v2_sha256),
        ("dataset_aggregate_sha256", &b.dataset_aggregate_sha256),
        ("trainer_binary_sha256", &b.trainer_binary_sha256),
    ] {
        if !is_sha256(v) {
            return Err(BindingError::Invalid(format!("{name} is not a sha256")));
        }
    }
    want("a2_sha256", &b.a2_sha256, exp.a2)?;
    want(
        "scope_clarification_sha256",
        &b.scope_clarification_sha256,
        exp.scope_clarification,
    )?;
    want("manifest_v2_sha256", &b.manifest_v2_sha256, exp.manifest_v2)?;
    want(
        "v1_v2_equivalence_report_sha256",
        &b.v1_v2_equivalence_report_sha256,
        exp.equivalence,
    )?;
    want(
        "dataset_aggregate_sha256",
        &b.dataset_aggregate_sha256,
        exp.dataset,
    )?;
    want(
        "row_reconciliation_sha256",
        &b.row_reconciliation_sha256,
        exp.reconciliation,
    )?;
    want(
        "training_split_sha256",
        &b.training_split_sha256,
        exp.training_split,
    )?;
    want(
        "strict_parser_commit",
        &b.strict_parser_commit,
        exp.strict_parser_commit,
    )?;
    want("trainer_commit", &b.trainer_commit, exp.trainer_commit)?;
    want(
        "trainer_binary_sha256",
        &b.trainer_binary_sha256,
        exp.trainer_binary,
    )?;
    Ok(b)
}
