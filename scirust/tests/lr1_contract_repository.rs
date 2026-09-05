//! Ensure the runtime semantic validator accepts the exact repository contract.
//!
//! The shell checker and Rust validator are intentionally independent guards;
//! this test prevents them from drifting while both appear green in isolation.

use scirust::lr1_contract::{self, TRAINING_CHUNKS, VALIDATION_CHUNKS};

const REPOSITORY_CONTRACT: &str = include_str!("../../integration/llama.cpp/lr1_contract_v1.json");

#[test]
fn repository_lr1_contract_matches_runtime_semantics() {
    let validated = lr1_contract::validate_text(REPOSITORY_CONTRACT)
        .expect("repository LR1 contract must satisfy the runtime validator");

    assert_eq!(validated.training_chunks, TRAINING_CHUNKS);
    assert_eq!(validated.validation_chunks, VALIDATION_CHUNKS);
}
