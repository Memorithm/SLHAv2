//! Static non-regression guards for LR1 evidence immutability.
//!
//! Stage-A negative or interrupted outcomes must remain inspectable rather than
//! being silently replaced by a later invocation using the same evidence path.

const TRAINER: &str = include_str!("../src/bin/slha_train_lr1.rs");
const EVALUATOR: &str = include_str!("../src/bin/slha_eval_rank.rs");

#[test]
fn trainer_refuses_to_overwrite_startup_evidence() {
    assert!(TRAINER
        .contains("let startup_manifest = format!(\"{}.startup.json\", args.training_manifest);"));
    assert!(TRAINER.contains("Path::new(&startup_manifest).exists()"));
    assert!(TRAINER.contains("refusing to overwrite a prior failed or incomplete attempt"));
}

#[test]
fn evaluator_refuses_to_overwrite_validation_evidence() {
    assert!(EVALUATOR.contains("Path::new(&args.output).exists()"));
    assert!(EVALUATOR.contains("refusing to overwrite prior validation evidence"));
}
