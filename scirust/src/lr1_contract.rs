//! Runtime validation for the frozen SLHA-LR1 Stage-A contract.
//!
//! CI guards the repository copy, but experiment binaries also accept a path.
//! They must therefore validate the supplied document semantically rather than
//! treating an arbitrary file hash as proof that the LR1 protocol was used.

use crate::json::Json;
use std::fs;

pub const CANDIDATE_ID: &str = "slha-lr1-pairwise-top16-all-layers-v1";
pub const MODEL_SHA256: &str = "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6";
pub const LLAMA_COMMIT: &str = "fdb1db877c526ec90f668eca1b858da5dba85560";
pub const SOURCE_REVISION: &str = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64";
pub const SOURCE_SHA256: &str = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4";
pub const TRAINING_CHUNKS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
pub const VALIDATION_CHUNKS: [usize; 4] = [13, 14, 15, 16];
pub const POPULATED_CHUNKS: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
pub const STORAGE_SLOTS: usize = 17;
pub const TOP_K: usize = 16;
pub const N_LAYERS: usize = 6;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedContract {
    pub source_revision: String,
    pub source_sha256: String,
    pub training_chunks: Vec<usize>,
    pub validation_chunks: Vec<usize>,
}

fn field<'a>(value: &'a Json, key: &str) -> Result<&'a Json, String> {
    value
        .get(key)
        .ok_or_else(|| format!("LR1_CONTRACT_MISSING_FIELD:{key}"))
}

fn object_field<'a>(value: &'a Json, key: &str) -> Result<&'a Json, String> {
    let child = field(value, key)?;
    match child {
        Json::Obj(_) => Ok(child),
        _ => Err(format!("LR1_CONTRACT_TYPE_MISMATCH:{key}:object")),
    }
}

fn string_field<'a>(value: &'a Json, key: &str) -> Result<&'a str, String> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("LR1_CONTRACT_TYPE_MISMATCH:{key}:string"))
}

fn bool_field(value: &Json, key: &str) -> Result<bool, String> {
    field(value, key)?
        .as_bool()
        .ok_or_else(|| format!("LR1_CONTRACT_TYPE_MISMATCH:{key}:bool"))
}

fn number_field(value: &Json, key: &str) -> Result<f64, String> {
    let number = field(value, key)?
        .as_f64()
        .ok_or_else(|| format!("LR1_CONTRACT_TYPE_MISMATCH:{key}:number"))?;
    if !number.is_finite() {
        return Err(format!("LR1_CONTRACT_NONFINITE:{key}"));
    }
    Ok(number)
}

fn exact_string(value: &Json, key: &str, expected: &str) -> Result<(), String> {
    let actual = string_field(value, key)?;
    if actual != expected {
        return Err(format!(
            "LR1_CONTRACT_DRIFT:{key}:expected={expected:?}:actual={actual:?}"
        ));
    }
    Ok(())
}

fn exact_bool(value: &Json, key: &str, expected: bool) -> Result<(), String> {
    let actual = bool_field(value, key)?;
    if actual != expected {
        return Err(format!(
            "LR1_CONTRACT_DRIFT:{key}:expected={expected}:actual={actual}"
        ));
    }
    Ok(())
}

fn exact_number(value: &Json, key: &str, expected: f64) -> Result<(), String> {
    let actual = number_field(value, key)?;
    if actual.to_bits() != expected.to_bits() {
        return Err(format!(
            "LR1_CONTRACT_DRIFT:{key}:expected={expected}:actual={actual}"
        ));
    }
    Ok(())
}

fn usize_array(value: &Json, key: &str) -> Result<Vec<usize>, String> {
    let array = field(value, key)?
        .as_array()
        .ok_or_else(|| format!("LR1_CONTRACT_TYPE_MISMATCH:{key}:array"))?;
    let mut out = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let number = item
            .as_f64()
            .ok_or_else(|| format!("LR1_CONTRACT_TYPE_MISMATCH:{key}[{index}]:number"))?;
        if !number.is_finite() || number < 0.0 || number.fract() != 0.0 || number > usize::MAX as f64 {
            return Err(format!("LR1_CONTRACT_INVALID_INTEGER:{key}[{index}]"));
        }
        out.push(number as usize);
    }
    Ok(out)
}

fn exact_usize(value: &Json, key: &str, expected: usize) -> Result<(), String> {
    let actual = number_field(value, key)?;
    if actual != expected as f64 {
        return Err(format!(
            "LR1_CONTRACT_DRIFT:{key}:expected={expected}:actual={actual}"
        ));
    }
    Ok(())
}

fn exact_usize_array(value: &Json, key: &str, expected: &[usize]) -> Result<Vec<usize>, String> {
    let actual = usize_array(value, key)?;
    if actual != expected {
        return Err(format!(
            "LR1_CONTRACT_DRIFT:{key}:expected={expected:?}:actual={actual:?}"
        ));
    }
    Ok(actual)
}

pub fn validate_file(path: &str) -> Result<ValidatedContract, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("LR1_CONTRACT_FILE:{path}:{e}"))?;
    if !metadata.file_type().is_file() {
        return Err(format!("LR1_CONTRACT_FILE_NOT_REGULAR:{path}"));
    }
    if metadata.len() > 64 * 1024 {
        return Err(format!("LR1_CONTRACT_FILE_TOO_LARGE:{}", metadata.len()));
    }
    let text = fs::read_to_string(path).map_err(|e| format!("LR1_CONTRACT_READ:{path}:{e}"))?;
    validate_text(&text)
}

pub fn validate_text(text: &str) -> Result<ValidatedContract, String> {
    let root = Json::parse_with_limit(text, 64 * 1024)
        .map_err(|e| format!("LR1_CONTRACT_JSON_INVALID:{e}"))?;
    if !matches!(root, Json::Obj(_)) {
        return Err("LR1_CONTRACT_ROOT_NOT_OBJECT".into());
    }

    exact_string(&root, "schema", "slha_lr1_contract_v1")?;
    exact_string(&root, "candidate_id", CANDIDATE_ID)?;
    exact_string(
        &root,
        "status",
        "PREREGISTERED_NON_PROTECTED_DEVELOPMENT_ONLY",
    )?;

    let model = object_field(&root, "model")?;
    exact_string(model, "name", "ggml-org/tiny-llamas/stories15M-q8_0.gguf")?;
    exact_string(model, "sha256", MODEL_SHA256)?;

    let llama = object_field(&root, "llama_cpp")?;
    exact_string(llama, "tag", "b9860")?;
    exact_string(llama, "commit", LLAMA_COMMIT)?;

    let candidate = object_field(&root, "candidate")?;
    exact_string(candidate, "objective", "pairwise-topk")?;
    exact_usize(candidate, "top_k", TOP_K)?;
    exact_string(candidate, "layers", "all")?;
    exact_number(candidate, "margin", 1.0)?;
    exact_usize(candidate, "max_negatives", 8)?;
    exact_string(candidate, "negative_policy", "boundary-then-seeded")?;
    exact_usize(candidate, "seed", 7)?;
    exact_number(candidate, "learning_rate", 1.0e-9)?;
    exact_usize(candidate, "epochs", 2)?;
    exact_usize(candidate, "steps", 0)?;
    exact_usize(candidate, "batch", 16)?;
    exact_usize(candidate, "max_keys", 256)?;
    exact_number(candidate, "geometry_weight", 0.25)?;
    exact_string(candidate, "codec", "mixed")?;
    exact_bool(candidate, "allow_hyperparameter_sweep", false)?;
    exact_bool(candidate, "allow_layer_subset_sweep", false)?;
    exact_bool(candidate, "allow_objective_sweep", false)?;

    let source = object_field(&root, "development_source")?;
    exact_string(source, "repository", "roneneldan/TinyStories")?;
    exact_string(source, "revision", SOURCE_REVISION)?;
    exact_string(source, "file", "TinyStories-valid.txt")?;
    exact_string(source, "source_sha256", SOURCE_SHA256)?;
    exact_string(source, "license", "cdla-sharing-1.0")?;
    let derivation = object_field(source, "derivation")?;
    exact_usize(derivation, "prefix_bytes", 262_144)?;
    exact_bool(derivation, "truncate_to_last_newline", true)?;
    exact_string(derivation, "encoding", "utf-8")?;
    exact_usize(source, "rank_dataset_evaluation_chunks", 16)?;
    exact_usize(source, "rank_dataset_storage_slots", STORAGE_SLOTS)?;

    let indexing = object_field(source, "chunk_indexing")?;
    exact_usize(indexing, "initial_storage_slot", 0)?;
    exact_bool(indexing, "initial_storage_slot_must_be_empty", true)?;
    exact_usize(indexing, "first_populated_chunk", 1)?;
    exact_usize(indexing, "last_populated_chunk", 16)?;

    let training_chunks = exact_usize_array(source, "training_chunks", &TRAINING_CHUNKS)?;
    let validation_chunks = exact_usize_array(source, "validation_chunks", &VALIDATION_CHUNKS)?;

    let holdout = object_field(&root, "protected_holdout")?;
    exact_string(
        holdout,
        "path",
        "integration/llama.cpp/fixtures/tinystories_synthetic_holdout.txt",
    )?;
    exact_string(
        holdout,
        "sha256",
        "eed5a2cebe9a23a12475fc86e0a3e23e0178adecd6d57aff443a58af599e4a11",
    )?;
    exact_string(holdout, "stage_a_access", "FORBIDDEN")?;

    let invariants = object_field(&root, "invariants")?;
    for key in [
        "baseline_logits_training_only",
        "baseline_logits_inference_forbidden",
        "external_k_oracle_knobs_forbidden",
        "initial_weights_must_be_hash_frozen_before_optimizer",
        "derived_development_corpus_must_be_hash_frozen_before_optimizer",
        "candidate_weights_must_be_hash_frozen_before_diagnostic_ppl",
        "protected_holdout_must_not_select_or_tune_candidate",
        "negative_result_must_be_retained",
    ] {
        exact_bool(invariants, key, true)?;
    }

    let stage_a = object_field(&root, "stage_a")?;
    exact_bool(stage_a, "protected_holdout_allowed", false)?;
    exact_string(
        stage_a,
        "training_short_context_policy",
        "retain_existing_pairwise_topk_geometry_semantics",
    )?;
    exact_string(stage_a, "ranking_metric_row_rule", "n_visible > top_k")?;
    exact_string(
        stage_a,
        "ranking_metric_short_rows",
        "excluded_from_topk_metrics_only",
    )?;

    let stage_b = object_field(&root, "stage_b")?;
    exact_bool(stage_b, "enabled_by_this_contract", false)?;
    exact_bool(stage_b, "require_separate_pre_holdout_freeze", true)?;

    Ok(ValidatedContract {
        source_revision: SOURCE_REVISION.to_owned(),
        source_sha256: SOURCE_SHA256.to_owned(),
        training_chunks,
        validation_chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_contract() -> String {
        format!(
            r#"{{
              "schema":"slha_lr1_contract_v1",
              "candidate_id":"{CANDIDATE_ID}",
              "status":"PREREGISTERED_NON_PROTECTED_DEVELOPMENT_ONLY",
              "model":{{"name":"ggml-org/tiny-llamas/stories15M-q8_0.gguf","sha256":"{MODEL_SHA256}"}},
              "llama_cpp":{{"tag":"b9860","commit":"{LLAMA_COMMIT}"}},
              "candidate":{{"objective":"pairwise-topk","top_k":16,"layers":"all","margin":1.0,"max_negatives":8,"negative_policy":"boundary-then-seeded","seed":7,"learning_rate":1e-9,"epochs":2,"steps":0,"batch":16,"max_keys":256,"geometry_weight":0.25,"codec":"mixed","allow_hyperparameter_sweep":false,"allow_layer_subset_sweep":false,"allow_objective_sweep":false}},
              "development_source":{{"repository":"roneneldan/TinyStories","revision":"{SOURCE_REVISION}","file":"TinyStories-valid.txt","source_sha256":"{SOURCE_SHA256}","license":"cdla-sharing-1.0","derivation":{{"prefix_bytes":262144,"truncate_to_last_newline":true,"encoding":"utf-8"}},"rank_dataset_evaluation_chunks":16,"rank_dataset_storage_slots":17,"chunk_indexing":{{"initial_storage_slot":0,"initial_storage_slot_must_be_empty":true,"first_populated_chunk":1,"last_populated_chunk":16,"reason":"structural"}},"training_chunks":[1,2,3,4,5,6,7,8,9,10,11,12],"validation_chunks":[13,14,15,16]}},
              "protected_holdout":{{"path":"integration/llama.cpp/fixtures/tinystories_synthetic_holdout.txt","sha256":"eed5a2cebe9a23a12475fc86e0a3e23e0178adecd6d57aff443a58af599e4a11","stage_a_access":"FORBIDDEN"}},
              "invariants":{{"baseline_logits_training_only":true,"baseline_logits_inference_forbidden":true,"external_k_oracle_knobs_forbidden":true,"initial_weights_must_be_hash_frozen_before_optimizer":true,"derived_development_corpus_must_be_hash_frozen_before_optimizer":true,"candidate_weights_must_be_hash_frozen_before_diagnostic_ppl":true,"protected_holdout_must_not_select_or_tune_candidate":true,"negative_result_must_be_retained":true}},
              "stage_a":{{"protected_holdout_allowed":false,"training_short_context_policy":"retain_existing_pairwise_topk_geometry_semantics","ranking_metric_row_rule":"n_visible > top_k","ranking_metric_short_rows":"excluded_from_topk_metrics_only"}},
              "stage_b":{{"enabled_by_this_contract":false,"require_separate_pre_holdout_freeze":true}}
            }}"#
        )
    }

    #[test]
    fn accepts_frozen_semantics() {
        let contract = validate_text(&valid_contract()).expect("valid LR1 contract");
        assert_eq!(contract.training_chunks, TRAINING_CHUNKS);
        assert_eq!(contract.validation_chunks, VALIDATION_CHUNKS);
    }

    #[test]
    fn rejects_tuning_drift() {
        let text = valid_contract().replace("\"top_k\":16", "\"top_k\":8");
        let error = validate_text(&text).unwrap_err();
        assert!(error.contains("top_k"));
    }

    #[test]
    fn rejects_split_drift() {
        let text = valid_contract().replace(
            "\"validation_chunks\":[13,14,15,16]",
            "\"validation_chunks\":[12,13,14,15]",
        );
        let error = validate_text(&text).unwrap_err();
        assert!(error.contains("validation_chunks"));
    }

    #[test]
    fn rejects_holdout_enablement() {
        let text = valid_contract().replace(
            "\"protected_holdout_allowed\":false",
            "\"protected_holdout_allowed\":true",
        );
        let error = validate_text(&text).unwrap_err();
        assert!(error.contains("protected_holdout_allowed"));
    }
}
