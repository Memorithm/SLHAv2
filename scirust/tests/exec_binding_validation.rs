//! Production-linked tests for the execution-binding validator.
//!
//! These drive `scirust::exec_binding::parse_and_verify`, the exact entry point
//! the trainer calls before any staging directory exists. Assertions check the
//! stable machine-readable code, not merely that validation failed.
//!
//! Requires the `serde` feature (the ranking-trainer toolchain).

#![cfg(feature = "serde")]

use scirust::exec_binding::{parse_and_verify, BindingError, ExecutionBinding, Expected};

const A2: &str = "18bae50ac9716f343912e8364a799b67fd0ce64bfed44569118b5f4a8d75f861";
const SCOPE: &str = "0ee924e2750d12396a996b5137ea917e8daccc1e7c279deaddf9bc426a92c5d9";
const MAN_V1: &str = "524652d063ec6282347e639958a161c45554121fcbf255ee5af4bbaf385699ce";
const MAN_V2: &str = "7212e34d31d5b38fa39467d992e9db790915b608c339ff59033dbe52e0c249ac";
const EQUIV: &str = "5acecbaeaf0d1311ab9afee834e07225bd70c0bacb64ccdec451a57abd12f590";
const AUDIT: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const DATASET: &str = "a52348fcc329a69d8f477a45068fb51da29b565ea8a5a1645e961d4ec6622025";
const RECON: &str = "85de6936a525cf6c77ec427a0d33ace434d153fb08f4ee4a824dec4da63961d6";
const SPLITM: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const SPLITT: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const SPLITV: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const SPLITD: &str = "5555555555555555555555555555555555555555555555555555555555555555";
const PARSER_COMMIT: &str = "d1810c603bf0475cef02790298cf1c37dcb71860";
const TRAINER_COMMIT: &str = "6666666666666666666666666666666666666666";
const TRAINER_BIN: &str = "7777777777777777777777777777777777777777777777777777777777777777";
const FILE_SHA: &str = "8888888888888888888888888888888888888888888888888888888888888888";

fn expected() -> Expected<'static> {
    Expected {
        a2: A2,
        scope_clarification: SCOPE,
        manifest_v2: MAN_V2,
        equivalence: EQUIV,
        dataset: DATASET,
        reconciliation: RECON,
        training_split: SPLITT,
        strict_parser_commit: PARSER_COMMIT,
        trainer_commit: TRAINER_COMMIT,
        trainer_binary: TRAINER_BIN,
    }
}

fn valid_json() -> String {
    format!(
        r#"{{
  "schema": "slha_a2_execution_binding_v2",
  "statement": "binds A2 to its concrete manifest-v2 execution input",
  "a2_sha256": "{A2}",
  "scope_clarification_sha256": "{SCOPE}",
  "manifest_v1_sha256": "{MAN_V1}",
  "manifest_v2_sha256": "{MAN_V2}",
  "v1_v2_equivalence_report_sha256": "{EQUIV}",
  "strict_manifest_parser_audit_sha256": "{AUDIT}",
  "dataset_aggregate_sha256": "{DATASET}",
  "row_reconciliation_sha256": "{RECON}",
  "split_manifest_sha256": "{SPLITM}",
  "training_split_sha256": "{SPLITT}",
  "ranking_validation_split_sha256": "{SPLITV}",
  "training_diagnostics_split_sha256": "{SPLITD}",
  "strict_parser_commit": "{PARSER_COMMIT}",
  "trainer_commit": "{TRAINER_COMMIT}",
  "trainer_binary_sha256": "{TRAINER_BIN}",
  "valid": true
}}"#
    )
}

fn run(text: &str) -> Result<ExecutionBinding, BindingError> {
    parse_and_verify(text.as_bytes(), FILE_SHA, FILE_SHA, &expected())
}

fn code(r: Result<ExecutionBinding, BindingError>) -> &'static str {
    match r {
        Ok(_) => "ACCEPTED",
        Err(e) => e.code(),
    }
}

/// Replace once, asserting the anchor exists so no test silently runs unmutated.
fn mutate(from: &str, to: &str) -> String {
    let s = valid_json();
    assert!(s.contains(from), "fixture does not contain {from:?}");
    s.replacen(from, to, 1)
}

#[test]
fn valid_binding_is_accepted() {
    let b = run(&valid_json()).expect("the valid binding must be accepted");
    assert!(b.valid);
    assert_eq!(b.trainer_binary_sha256, TRAINER_BIN);
}

#[test]
fn invalid_expected_sha_syntax_is_rejected() {
    let r = parse_and_verify(valid_json().as_bytes(), FILE_SHA, "not-a-sha", &expected());
    assert_eq!(code(r), "EXEC_BINDING_HASH_INVALID");
}

#[test]
fn file_hash_mismatch_is_rejected() {
    let other = "9".repeat(64);
    let r = parse_and_verify(valid_json().as_bytes(), &other, FILE_SHA, &expected());
    assert_eq!(code(r), "EXEC_BINDING_HASH_MISMATCH");
}

#[test]
fn trailing_non_whitespace_is_rejected() {
    let text = format!("{}trailing", valid_json());
    assert_eq!(code(run(&text)), "EXEC_BINDING_TRAILING_DATA");
}

#[test]
fn trailing_whitespace_is_allowed() {
    let text = format!("{}\n\n  ", valid_json());
    assert!(run(&text).is_ok());
}

#[test]
fn duplicate_top_level_field_is_rejected() {
    let text = mutate(
        "\"valid\": true",
        "\"a2_sha256\": \"x\",\n  \"valid\": true",
    );
    assert_eq!(code(run(&text)), "EXEC_BINDING_DUPLICATE_FIELD");
}

#[test]
fn unknown_top_level_field_is_rejected() {
    let text = mutate("\"valid\": true", "\"surprise\": 1,\n  \"valid\": true");
    assert_eq!(code(run(&text)), "EXEC_BINDING_UNKNOWN_FIELD");
}

#[test]
fn nested_same_name_object_is_rejected() {
    let text = mutate(
        "\"valid\": true",
        "\"note\": {\"a2_sha256\": \"x\"},\n  \"valid\": true",
    );
    assert_eq!(code(run(&text)), "EXEC_BINDING_UNKNOWN_FIELD");
}

#[test]
fn wrong_json_type_is_rejected() {
    let text = mutate("\"valid\": true", "\"valid\": \"yes\"");
    assert_eq!(code(run(&text)), "EXEC_BINDING_JSON_INVALID");
}

#[test]
fn missing_required_field_is_rejected() {
    let text = valid_json().replacen(
        &format!("  \"row_reconciliation_sha256\": \"{RECON}\",\n"),
        "",
        1,
    );
    assert_eq!(code(run(&text)), "EXEC_BINDING_JSON_INVALID");
}

#[test]
fn truncated_json_is_rejected() {
    let s = valid_json();
    assert_eq!(code(run(&s[..s.len() / 2])), "EXEC_BINDING_JSON_INVALID");
}

#[test]
fn unsupported_schema_is_rejected() {
    let text = mutate(
        "slha_a2_execution_binding_v2",
        "slha_a2_execution_binding_v9",
    );
    assert_eq!(code(run(&text)), "EXEC_BINDING_SCHEMA_UNSUPPORTED");
}

#[test]
fn valid_false_is_rejected() {
    let text = mutate("\"valid\": true", "\"valid\": false");
    assert_eq!(code(run(&text)), "EXEC_BINDING_INVALID");
}

#[test]
fn a2_mismatch_is_rejected() {
    let text = mutate(A2, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_A2_MISMATCH");
}

#[test]
fn scope_clarification_mismatch_is_rejected() {
    let text = mutate(SCOPE, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_SCOPE_MISMATCH");
}

#[test]
fn manifest_v2_mismatch_is_rejected() {
    let text = mutate(MAN_V2, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_MANIFEST_MISMATCH");
}

#[test]
fn equivalence_report_mismatch_is_rejected() {
    let text = mutate(EQUIV, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_EQUIVALENCE_MISMATCH");
}

#[test]
fn dataset_mismatch_is_rejected() {
    let text = mutate(DATASET, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_DATASET_MISMATCH");
}

#[test]
fn row_reconciliation_mismatch_is_rejected() {
    let text = mutate(RECON, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_RECONCILIATION_MISMATCH");
}

#[test]
fn training_split_mismatch_is_rejected() {
    let text = mutate(SPLITT, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_SPLIT_MISMATCH");
}

#[test]
fn parser_commit_mismatch_is_rejected() {
    let text = mutate(PARSER_COMMIT, "0000000000000000000000000000000000000000");
    assert_eq!(code(run(&text)), "EXEC_BINDING_PARSER_COMMIT_MISMATCH");
}

#[test]
fn trainer_commit_mismatch_is_rejected() {
    let text = mutate(TRAINER_COMMIT, "0000000000000000000000000000000000000000");
    assert_eq!(code(run(&text)), "EXEC_BINDING_TRAINER_COMMIT_MISMATCH");
}

#[test]
fn trainer_binary_mismatch_is_rejected() {
    let text = mutate(TRAINER_BIN, &"0".repeat(64));
    assert_eq!(code(run(&text)), "EXEC_BINDING_TRAINER_BINARY_MISMATCH");
}

#[test]
fn non_sha256_hash_field_is_rejected() {
    let text = mutate(
        &format!("\"a2_sha256\": \"{A2}\""),
        "\"a2_sha256\": \"short\"",
    );
    assert_eq!(code(run(&text)), "EXEC_BINDING_INVALID");
}

#[test]
fn repeated_verification_is_stable() {
    let t = valid_json();
    assert_eq!(code(run(&t)), "ACCEPTED");
    assert_eq!(code(run(&t)), "ACCEPTED");
}
