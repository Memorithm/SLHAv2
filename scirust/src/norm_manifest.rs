//! Strict, structural parsing of the frozen L2 normalisation manifest.
//!
//! An earlier loader resolved fields by textual `str::find`. A malformed-input
//! audit proved that unsafe: it accepted a nested field with the same name, data
//! after the closing brace, and a duplicated top-level key, because substring
//! search cannot see JSON structure. This module replaces it with typed
//! deserialization through `serde_json`, which consumes the complete byte
//! stream, rejects duplicate struct fields, and rejects unknown fields.
//!
//! Array order is NOT semantic. A manifest whose layer entries are unsorted is
//! valid provided the set is exactly 0..=27 with unique IDs; records are
//! canonicalised by layer ID after parsing.

use serde::Deserialize;
use std::collections::BTreeMap;

/// Machine-readable rejection codes. Audits assert the exact code, not merely a
/// non-zero exit status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    FileNotFound(String),
    HashMismatch {
        expected: String,
        actual: String,
    },
    JsonInvalid(String),
    TrailingData(String),
    DuplicateField(String),
    UnknownField(String),
    SchemaUnsupported(String),
    BindingMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    LayerDuplicate(usize),
    LayerSetInvalid(String),
    ScaleInvalid {
        layer: usize,
        what: String,
    },
    ScaleBitsMismatch {
        layer: usize,
        what: String,
    },
    EpsilonMismatch {
        layer: usize,
    },
}

impl ManifestError {
    pub fn code(&self) -> &'static str {
        match self {
            ManifestError::FileNotFound(_) => "MANIFEST_FILE_NOT_FOUND",
            ManifestError::HashMismatch { .. } => "MANIFEST_HASH_MISMATCH",
            ManifestError::JsonInvalid(_) => "MANIFEST_JSON_INVALID",
            ManifestError::TrailingData(_) => "MANIFEST_TRAILING_DATA",
            ManifestError::DuplicateField(_) => "MANIFEST_DUPLICATE_FIELD",
            ManifestError::UnknownField(_) => "MANIFEST_UNKNOWN_FIELD",
            ManifestError::SchemaUnsupported(_) => "MANIFEST_SCHEMA_UNSUPPORTED",
            ManifestError::BindingMismatch { .. } => "MANIFEST_BINDING_MISMATCH",
            ManifestError::LayerDuplicate(_) => "MANIFEST_LAYER_DUPLICATE",
            ManifestError::LayerSetInvalid(_) => "MANIFEST_LAYER_SET_INVALID",
            ManifestError::ScaleInvalid { .. } => "MANIFEST_SCALE_INVALID",
            ManifestError::ScaleBitsMismatch { .. } => "MANIFEST_SCALE_BITS_MISMATCH",
            ManifestError::EpsilonMismatch { .. } => "MANIFEST_EPSILON_MISMATCH",
        }
    }
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:?}", self.code(), self)
    }
}

pub const EXPECTED_SCHEMA: &str = "slha_l2_normalisation_manifest_v2";
pub const N_LAYERS: usize = 28;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bindings {
    pub preregistration_a2_sha256: String,
    pub scope_clarification_sha256: String,
    pub dataset_aggregate_sha256: String,
    pub row_reconciliation_sha256: String,
    pub split_manifest_sha256: String,
    pub training_split_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRecord {
    pub layer: usize,
    pub training_rows: u64,
    pub training_active_scores: u64,
    pub sum_squared_baseline_scores: f64,
    pub rms_b: f64,
    pub rms_b_squared_plus_epsilon: f64,
    pub epsilon: f64,
    pub finite: bool,
    pub positive: bool,
    pub rms_b_decimal: String,
    pub rms_b_f64_bits: u64,
    pub rms_b_f64_hex: String,
    pub scale_decimal: String,
    pub scale_f64_bits: u64,
    pub scale_f64_hex: String,
    pub epsilon_decimal: String,
    pub epsilon_f64_bits: u64,
}

/// Top-level manifest. Unknown fields are rejected, and `serde` rejects a
/// duplicated field for any of these names.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub rule: String,
    pub source_split: String,
    pub forbidden_sources: Vec<String>,
    pub accumulation: String,
    pub max_keys_per_row: u64,
    pub epsilon: f64,
    pub source_split_hash: String,
    pub accumulation_implementation_hash: String,
    pub objective_implementation_hash: String,
    pub supersedes: serde_json::Value,
    pub binds: Bindings,
    pub layers: Vec<LayerRecord>,
    pub checks: serde_json::Value,
    pub valid: bool,
}

/// A verified, canonicalised set of frozen layer scales.
pub struct FrozenScales {
    pub by_layer: BTreeMap<usize, (u64, u64, u64)>, // layer -> (rms_bits, scale_bits, eps_bits)
    pub manifest_sha256: String,
}

impl FrozenScales {
    /// Ordered RMS bit vector, layer 0..N-1, for the startup lineage hash.
    pub fn ordered_rms_bits(&self) -> Vec<u64> {
        self.by_layer.values().map(|v| v.0).collect()
    }
}

fn binding(field: &str, actual: &str, expected: &str) -> Result<(), ManifestError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ManifestError::BindingMismatch {
            field: field.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

/// Parse and fully validate the manifest bytes.
///
/// `expected_*` are the frozen bindings the caller requires. The complete byte
/// stream must be consumed: data after the closing brace is a rejection.
pub fn parse_and_verify(
    bytes: &[u8],
    manifest_sha256: &str,
    expected_a2: &str,
    expected_dataset: &str,
    expected_recon: &str,
    expected_split: Option<&str>,
    expected_epsilon: f64,
) -> Result<FrozenScales, ManifestError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| ManifestError::JsonInvalid(format!("not UTF-8: {e}")))?;

    // Typed deserialization over the WHOLE stream. `end()` is what rejects
    // trailing non-whitespace data; a plain from_str would silently stop at the
    // closing brace.
    let mut de = serde_json::Deserializer::from_str(text);
    let m = Manifest::deserialize(&mut de).map_err(|e| {
        let s = e.to_string();
        if s.contains("duplicate field") {
            ManifestError::DuplicateField(s)
        } else if s.contains("unknown field") {
            ManifestError::UnknownField(s)
        } else {
            ManifestError::JsonInvalid(s)
        }
    })?;
    de.end()
        .map_err(|e| ManifestError::TrailingData(e.to_string()))?;

    if m.schema != EXPECTED_SCHEMA {
        return Err(ManifestError::SchemaUnsupported(m.schema));
    }
    if !m.valid {
        return Err(ManifestError::SchemaUnsupported("valid != true".into()));
    }
    binding(
        "preregistration_a2_sha256",
        &m.binds.preregistration_a2_sha256,
        expected_a2,
    )?;
    binding(
        "dataset_aggregate_sha256",
        &m.binds.dataset_aggregate_sha256,
        expected_dataset,
    )?;
    binding(
        "row_reconciliation_sha256",
        &m.binds.row_reconciliation_sha256,
        expected_recon,
    )?;
    if let Some(sp) = expected_split {
        binding("training_split_sha256", &m.binds.training_split_sha256, sp)?;
    }
    if m.epsilon.to_bits() != expected_epsilon.to_bits() {
        return Err(ManifestError::EpsilonMismatch { layer: usize::MAX });
    }
    if m.layers.len() != N_LAYERS {
        return Err(ManifestError::LayerSetInvalid(format!(
            "{} layer records, expected {N_LAYERS}",
            m.layers.len()
        )));
    }

    let mut by_layer: BTreeMap<usize, (u64, u64, u64)> = BTreeMap::new();
    for r in &m.layers {
        if r.layer >= N_LAYERS {
            return Err(ManifestError::LayerSetInvalid(format!(
                "layer {} is outside 0..{}",
                r.layer,
                N_LAYERS - 1
            )));
        }
        if by_layer.contains_key(&r.layer) {
            return Err(ManifestError::LayerDuplicate(r.layer));
        }
        let bad = |what: &str| ManifestError::ScaleInvalid {
            layer: r.layer,
            what: what.to_string(),
        };
        if r.training_rows == 0 {
            return Err(bad("training_rows is zero"));
        }
        if r.training_active_scores == 0 {
            return Err(bad("active-score count is zero"));
        }
        if !r.sum_squared_baseline_scores.is_finite() || r.sum_squared_baseline_scores < 0.0 {
            return Err(bad("sum of squares is not finite and non-negative"));
        }
        let rms = f64::from_bits(r.rms_b_f64_bits);
        let scale = f64::from_bits(r.scale_f64_bits);
        if !rms.is_finite() || rms <= 0.0 {
            return Err(bad("rms_b is not positive-finite"));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(bad("scale is not positive-finite"));
        }
        // the decimal spelling must round-trip to the declared bit pattern
        if r.rms_b_decimal.parse::<f64>().map(|v| v.to_bits()) != Ok(r.rms_b_f64_bits) {
            return Err(ManifestError::ScaleBitsMismatch {
                layer: r.layer,
                what: "rms_b_decimal".into(),
            });
        }
        if r.scale_decimal.parse::<f64>().map(|v| v.to_bits()) != Ok(r.scale_f64_bits) {
            return Err(ManifestError::ScaleBitsMismatch {
                layer: r.layer,
                what: "scale_decimal".into(),
            });
        }
        if r.epsilon_decimal.parse::<f64>().map(|v| v.to_bits()) != Ok(r.epsilon_f64_bits) {
            return Err(ManifestError::ScaleBitsMismatch {
                layer: r.layer,
                what: "epsilon_decimal".into(),
            });
        }
        if r.epsilon_f64_bits != expected_epsilon.to_bits() {
            return Err(ManifestError::EpsilonMismatch { layer: r.layer });
        }
        by_layer.insert(
            r.layer,
            (r.rms_b_f64_bits, r.scale_f64_bits, r.epsilon_f64_bits),
        );
    }
    // canonicalise: the set must be exactly 0..=27; input order is not semantic
    let ids: Vec<usize> = by_layer.keys().copied().collect();
    if ids != (0..N_LAYERS).collect::<Vec<_>>() {
        return Err(ManifestError::LayerSetInvalid(format!(
            "layer IDs are {ids:?}, expected 0..{}",
            N_LAYERS - 1
        )));
    }
    Ok(FrozenScales {
        by_layer,
        manifest_sha256: manifest_sha256.to_string(),
    })
}
