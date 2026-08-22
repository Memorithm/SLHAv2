//! Production-linked tests for the strict normalisation-manifest parser.
//!
//! These drive `scirust::norm_manifest::parse_and_verify` — the exact entry
//! point the trainer calls — against the real frozen manifest and mutations of
//! it. They exist because an earlier substring-based loader passed an ad-hoc
//! shell audit while silently accepting a nested same-name field, trailing data
//! after the closing brace, and a duplicated top-level key. A shell harness is
//! not a substitute for committed tests.
//!
//! Requires the `serde` feature (the ranking-trainer toolchain).

#![cfg(feature = "serde")]

use scirust::norm_manifest::{parse_and_verify, ManifestError, N_LAYERS};

const FIXTURE: &str = include_str!("data/norm_manifest_v2.json");

const A2: &str = "18bae50ac9716f343912e8364a799b67fd0ce64bfed44569118b5f4a8d75f861";
const DATASET: &str = "a52348fcc329a69d8f477a45068fb51da29b565ea8a5a1645e961d4ec6622025";
const RECON: &str = "85de6936a525cf6c77ec427a0d33ace434d153fb08f4ee4a824dec4da63961d6";
const EPS: f64 = 1.0e-6;

fn parse(text: &str) -> Result<scirust::norm_manifest::FrozenScales, ManifestError> {
    parse_and_verify(
        text.as_bytes(),
        "sha-not-checked-here",
        A2,
        DATASET,
        RECON,
        None,
        EPS,
    )
}

/// Replace the first occurrence of `from` with `to`, asserting it was present so
/// a test can never silently exercise an unmutated fixture.
fn mutate(from: &str, to: &str) -> String {
    assert!(FIXTURE.contains(from), "fixture does not contain {from:?}");
    FIXTURE.replacen(from, to, 1)
}

fn code(r: Result<scirust::norm_manifest::FrozenScales, ManifestError>) -> &'static str {
    match r {
        Ok(_) => "ACCEPTED",
        Err(e) => e.code(),
    }
}

#[test]
fn valid_manifest_is_accepted_with_all_layers() {
    let s = parse(FIXTURE).expect("the frozen manifest must parse");
    assert_eq!(s.by_layer.len(), N_LAYERS);
    assert_eq!(s.ordered_rms_bits().len(), N_LAYERS);
}

#[test]
fn exact_f64_bits_are_loaded_not_rounded_decimals() {
    let s = parse(FIXTURE).unwrap();
    // layer 0 of the frozen production manifest
    let (rms_bits, scale_bits, eps_bits) = s.by_layer[&0];
    assert_eq!(rms_bits, 0x40e1_a16c_60b6_828a, "exact rms bit pattern");
    assert_eq!(f64::from_bits(rms_bits), 36107.38680577751);
    assert!(f64::from_bits(scale_bits) > 0.0);
    assert_eq!(f64::from_bits(eps_bits), EPS);
}

#[test]
fn ordered_bits_are_canonical_by_layer_id() {
    let s = parse(FIXTURE).unwrap();
    let ids: Vec<usize> = s.by_layer.keys().copied().collect();
    assert_eq!(ids, (0..N_LAYERS).collect::<Vec<_>>());
}

#[test]
fn reordered_layers_are_accepted_and_canonicalised() {
    // Array order is NOT semantic. Reversing the layer array must still yield
    // the identical canonical ordered bit vector.
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut o = v.clone();
    let mut layers = o["layers"].as_array().unwrap().clone();
    layers.reverse();
    o["layers"] = serde_json::Value::Array(layers);
    let text = serde_json::to_string_pretty(&o).unwrap();
    let a = parse(&text).expect("a reordered complete layer set is valid");
    let b = parse(FIXTURE).unwrap();
    assert_eq!(a.ordered_rms_bits(), b.ordered_rms_bits());
}

#[test]
fn trailing_non_whitespace_is_rejected() {
    let text = format!("{FIXTURE}trailing-garbage");
    assert_eq!(code(parse(&text)), "MANIFEST_TRAILING_DATA");
}

#[test]
fn trailing_whitespace_is_allowed() {
    let text = format!("{FIXTURE}\n\n   \n");
    assert!(parse(&text).is_ok());
}

#[test]
fn duplicate_top_level_field_is_rejected() {
    let text = mutate("\"layers\":", "\"schema\": \"evil\",\n \"layers\":");
    assert_eq!(code(parse(&text)), "MANIFEST_DUPLICATE_FIELD");
}

#[test]
fn unknown_top_level_field_is_rejected() {
    let text = mutate("\"layers\":", "\"surprise\": 1,\n \"layers\":");
    assert_eq!(code(parse(&text)), "MANIFEST_UNKNOWN_FIELD");
}

#[test]
fn unknown_layer_field_is_rejected() {
    let text = mutate("\"layer\": 0,", "\"layer\": 0,\n   \"surprise\": 1,");
    assert_eq!(code(parse(&text)), "MANIFEST_UNKNOWN_FIELD");
}

#[test]
fn nested_same_name_object_is_rejected() {
    // The exact case the substring extractor accepted: a nested object carrying
    // a field name the loader also looks for at layer level.
    let text = mutate(
        "\"layer\": 0,",
        "\"layer\": 0,\n   \"note\": {\"rms_b_f64_bits\": 1},",
    );
    assert_eq!(code(parse(&text)), "MANIFEST_UNKNOWN_FIELD");
}

#[test]
fn unknown_bindings_field_is_rejected() {
    let text = mutate(
        "\"preregistration_a2_sha256\":",
        "\"surprise\": 1,\n   \"preregistration_a2_sha256\":",
    );
    assert_eq!(code(parse(&text)), "MANIFEST_UNKNOWN_FIELD");
}

#[test]
fn truncated_json_is_rejected() {
    let text = &FIXTURE[..FIXTURE.len() / 2];
    assert_eq!(code(parse(text)), "MANIFEST_JSON_INVALID");
}

#[test]
fn wrong_json_type_is_rejected() {
    let text = mutate(
        "\"rms_b_f64_bits\": ",
        "\"rms_b_f64_bits\": \"not-a-number\", \"x\": ",
    );
    assert!(matches!(
        code(parse(&text)),
        "MANIFEST_JSON_INVALID" | "MANIFEST_UNKNOWN_FIELD"
    ));
}

#[test]
fn missing_required_field_is_rejected() {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut o = v.clone();
    o.as_object_mut().unwrap().remove("epsilon");
    let text = serde_json::to_string(&o).unwrap();
    assert_eq!(code(parse(&text)), "MANIFEST_JSON_INVALID");
}

#[test]
fn unsupported_schema_is_rejected() {
    let text = mutate(
        "slha_l2_normalisation_manifest_v2",
        "slha_l2_normalisation_manifest_v9",
    );
    assert_eq!(code(parse(&text)), "MANIFEST_SCHEMA_UNSUPPORTED");
}

#[test]
fn valid_false_is_rejected() {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut o = v.clone();
    o["valid"] = serde_json::Value::Bool(false);
    let text = serde_json::to_string(&o).unwrap();
    assert_eq!(code(parse(&text)), "MANIFEST_SCHEMA_UNSUPPORTED");
}

fn with_layer<F: Fn(&mut serde_json::Value)>(idx: usize, f: F) -> String {
    let mut o: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    f(&mut o["layers"][idx]);
    serde_json::to_string(&o).unwrap()
}

#[test]
fn duplicate_layer_id_is_rejected() {
    let text = with_layer(5, |l| l["layer"] = serde_json::json!(4));
    assert_eq!(code(parse(&text)), "MANIFEST_LAYER_DUPLICATE");
}

#[test]
fn missing_layer_is_rejected() {
    let mut o: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    o["layers"].as_array_mut().unwrap().truncate(N_LAYERS - 1);
    let text = serde_json::to_string(&o).unwrap();
    assert_eq!(code(parse(&text)), "MANIFEST_LAYER_SET_INVALID");
}

#[test]
fn extra_unique_layer_is_rejected() {
    let mut o: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut extra = o["layers"][0].clone();
    extra["layer"] = serde_json::json!(N_LAYERS);
    o["layers"].as_array_mut().unwrap().push(extra);
    let text = serde_json::to_string(&o).unwrap();
    assert_eq!(code(parse(&text)), "MANIFEST_LAYER_SET_INVALID");
}

#[test]
fn zero_rms_is_rejected() {
    let text = with_layer(3, |l| {
        l["rms_b_f64_bits"] = serde_json::json!(0u64);
        l["rms_b_decimal"] = serde_json::json!("0.0");
    });
    assert_eq!(code(parse(&text)), "MANIFEST_SCALE_INVALID");
}

#[test]
fn negative_rms_is_rejected() {
    let bits = (-1.0f64).to_bits();
    let text = with_layer(3, |l| {
        l["rms_b_f64_bits"] = serde_json::json!(bits);
        l["rms_b_decimal"] = serde_json::json!("-1.0");
    });
    assert_eq!(code(parse(&text)), "MANIFEST_SCALE_INVALID");
}

#[test]
fn nan_bit_pattern_is_rejected() {
    let bits = f64::NAN.to_bits();
    let text = with_layer(7, |l| {
        l["rms_b_f64_bits"] = serde_json::json!(bits);
        l["rms_b_decimal"] = serde_json::json!("NaN");
    });
    assert_eq!(code(parse(&text)), "MANIFEST_SCALE_INVALID");
}

#[test]
fn infinite_bit_pattern_is_rejected() {
    let bits = f64::INFINITY.to_bits();
    let text = with_layer(7, |l| {
        l["rms_b_f64_bits"] = serde_json::json!(bits);
        l["rms_b_decimal"] = serde_json::json!("inf");
    });
    assert_eq!(code(parse(&text)), "MANIFEST_SCALE_INVALID");
}

#[test]
fn decimal_bit_mismatch_is_rejected() {
    let text = with_layer(2, |l| l["rms_b_decimal"] = serde_json::json!("1.0"));
    assert_eq!(code(parse(&text)), "MANIFEST_SCALE_BITS_MISMATCH");
}

#[test]
fn epsilon_mismatch_is_rejected() {
    let text = with_layer(9, |l| {
        l["epsilon_f64_bits"] = serde_json::json!(1u64);
        l["epsilon_decimal"] = serde_json::json!("5e-324");
    });
    assert_eq!(code(parse(&text)), "MANIFEST_EPSILON_MISMATCH");
}

#[test]
fn zero_active_score_count_is_rejected() {
    let text = with_layer(11, |l| {
        l["training_active_scores"] = serde_json::json!(0u64)
    });
    assert_eq!(code(parse(&text)), "MANIFEST_SCALE_INVALID");
}

#[test]
fn wrong_a2_binding_is_rejected() {
    let r = parse_and_verify(
        FIXTURE.as_bytes(),
        "x",
        &"0".repeat(64),
        DATASET,
        RECON,
        None,
        EPS,
    );
    assert_eq!(code(r), "MANIFEST_BINDING_MISMATCH");
}

#[test]
fn wrong_dataset_binding_is_rejected() {
    let r = parse_and_verify(
        FIXTURE.as_bytes(),
        "x",
        A2,
        &"0".repeat(64),
        RECON,
        None,
        EPS,
    );
    assert_eq!(code(r), "MANIFEST_BINDING_MISMATCH");
}

#[test]
fn wrong_split_binding_is_rejected() {
    let bad = "0".repeat(64);
    let r = parse_and_verify(FIXTURE.as_bytes(), "x", A2, DATASET, RECON, Some(&bad), EPS);
    assert_eq!(code(r), "MANIFEST_BINDING_MISMATCH");
}

#[test]
fn wrong_reconciliation_binding_is_rejected() {
    let r = parse_and_verify(
        FIXTURE.as_bytes(),
        "x",
        A2,
        DATASET,
        &"0".repeat(64),
        None,
        EPS,
    );
    assert_eq!(code(r), "MANIFEST_BINDING_MISMATCH");
}

#[test]
fn caller_epsilon_mismatch_is_rejected() {
    let r = parse_and_verify(FIXTURE.as_bytes(), "x", A2, DATASET, RECON, None, 1.0e-5);
    assert_eq!(code(r), "MANIFEST_EPSILON_MISMATCH");
}

#[test]
fn repeated_parsing_yields_identical_scale_bits() {
    let a = parse(FIXTURE).unwrap().ordered_rms_bits();
    let b = parse(FIXTURE).unwrap().ordered_rms_bits();
    assert_eq!(a, b);
}
