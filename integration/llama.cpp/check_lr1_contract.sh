#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CONTRACT="$SCRIPT_DIR/lr1_contract_v1.json"

[[ -s "$CONTRACT" ]] || { echo "ERROR: missing LR1 contract: $CONTRACT" >&2; exit 2; }

python3 - "$CONTRACT" "$REPO_ROOT" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

contract_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])

with contract_path.open(encoding="utf-8") as f:
    c = json.load(f)

if c.get("schema") != "slha_lr1_contract_v1":
    raise SystemExit("LR1_CONTRACT_SCHEMA_MISMATCH")
if c.get("candidate_id") != "slha-lr1-pairwise-top16-all-layers-v1":
    raise SystemExit("LR1_CANDIDATE_ID_DRIFT")
if c.get("status") != "PREREGISTERED_NON_PROTECTED_DEVELOPMENT_ONLY":
    raise SystemExit("LR1_STATUS_DRIFT")

expected_candidate = {
    "objective": "pairwise-topk",
    "top_k": 16,
    "layers": "all",
    "margin": 1.0,
    "max_negatives": 8,
    "negative_policy": "boundary-then-seeded",
    "seed": 7,
    "learning_rate": 1e-9,
    "epochs": 2,
    "steps": 0,
    "batch": 16,
    "max_keys": 256,
    "geometry_weight": 0.25,
    "codec": "mixed",
    "allow_hyperparameter_sweep": False,
    "allow_layer_subset_sweep": False,
    "allow_objective_sweep": False,
}
if c.get("candidate") != expected_candidate:
    raise SystemExit("LR1_FROZEN_CANDIDATE_DRIFT")

source = c.get("development_source") or {}
if source.get("repository") != "roneneldan/TinyStories":
    raise SystemExit("LR1_DEVELOPMENT_SOURCE_DRIFT")
if source.get("revision") != "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64":
    raise SystemExit("LR1_DEVELOPMENT_REVISION_DRIFT")
if source.get("file") != "TinyStories-valid.txt":
    raise SystemExit("LR1_DEVELOPMENT_FILE_DRIFT")
if source.get("source_sha256") != "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4":
    raise SystemExit("LR1_DEVELOPMENT_SOURCE_HASH_DRIFT")
if source.get("derivation") != {
    "prefix_bytes": 262144,
    "truncate_to_last_newline": True,
    "encoding": "utf-8",
}:
    raise SystemExit("LR1_DEVELOPMENT_DERIVATION_DRIFT")

if source.get("rank_dataset_evaluation_chunks") != 16:
    raise SystemExit("LR1_EVALUATION_CHUNK_COUNT_DRIFT")
if source.get("rank_dataset_storage_slots") != 17:
    raise SystemExit("LR1_STORAGE_SLOT_COUNT_DRIFT")
expected_indexing = {
    "initial_storage_slot": 0,
    "initial_storage_slot_must_be_empty": True,
    "first_populated_chunk": 1,
    "last_populated_chunk": 16,
    "reason": "Pinned llama.cpp perplexity clears memory before each evaluation chunk; the SLHA clear hook increments the collector chunk before the first decode.",
}
if source.get("chunk_indexing") != expected_indexing:
    raise SystemExit("LR1_CHUNK_INDEXING_DRIFT")

train = source.get("training_chunks")
valid = source.get("validation_chunks")
expected_train = list(range(1, 13))
expected_valid = list(range(13, 17))
expected_populated = list(range(1, 17))
if train != expected_train or valid != expected_valid:
    raise SystemExit("LR1_SPLIT_DRIFT")
if set(train) & set(valid):
    raise SystemExit("LR1_SPLIT_OVERLAP")
if sorted(train + valid) != expected_populated:
    raise SystemExit("LR1_SPLIT_COVERAGE_INVALID")

hex64 = re.compile(r"^[0-9a-f]{64}$")
hex40 = re.compile(r"^[0-9a-f]{40}$")
if not hex40.fullmatch(source.get("revision", "")):
    raise SystemExit("LR1_SOURCE_REVISION_INVALID")
if not hex64.fullmatch(source.get("source_sha256", "")):
    raise SystemExit("LR1_SOURCE_HASH_INVALID")


def file_sha256(rel):
    p = root / rel
    if not p.is_file():
        raise SystemExit(f"LR1_REQUIRED_FILE_MISSING:{rel}")
    return hashlib.sha256(p.read_bytes()).hexdigest()

rank_diag = c.get("diagnostic_corpus") or {}
holdout = c.get("protected_holdout") or {}
if file_sha256(rank_diag.get("path", "")) != rank_diag.get("sha256"):
    raise SystemExit("LR1_DIAGNOSTIC_HASH_MISMATCH")
if file_sha256(holdout.get("path", "")) != holdout.get("sha256"):
    raise SystemExit("LR1_HOLDOUT_HASH_MISMATCH")
if rank_diag.get("sha256") == holdout.get("sha256"):
    raise SystemExit("LR1_DIAGNOSTIC_EQUALS_HOLDOUT")
if holdout.get("stage_a_access") != "FORBIDDEN":
    raise SystemExit("LR1_HOLDOUT_STAGE_A_NOT_FORBIDDEN")

stage_a = c.get("stage_a") or {}
if stage_a.get("protected_holdout_allowed") is not False:
    raise SystemExit("LR1_STAGE_A_HOLDOUT_ENABLED")
expected_short_context = {
    "training_short_context_policy": "retain_existing_pairwise_topk_geometry_semantics",
    "ranking_metric_row_rule": "n_visible > top_k",
    "ranking_metric_short_rows": "excluded_from_topk_metrics_only",
}
for key, value in expected_short_context.items():
    if stage_a.get(key) != value:
        raise SystemExit(f"LR1_SHORT_CONTEXT_CONTRACT_DRIFT:{key}")

if (c.get("stage_b") or {}).get("enabled_by_this_contract") is not False:
    raise SystemExit("LR1_STAGE_B_PREMATURELY_ENABLED")
if (c.get("stage_b") or {}).get("require_separate_pre_holdout_freeze") is not True:
    raise SystemExit("LR1_STAGE_B_FREEZE_GUARD_MISSING")

invariants = c.get("invariants") or {}
required_true = [
    "baseline_logits_training_only",
    "baseline_logits_inference_forbidden",
    "external_k_oracle_knobs_forbidden",
    "initial_weights_must_be_hash_frozen_before_optimizer",
    "derived_development_corpus_must_be_hash_frozen_before_optimizer",
    "candidate_weights_must_be_hash_frozen_before_diagnostic_ppl",
    "protected_holdout_must_not_select_or_tune_candidate",
    "negative_result_must_be_retained",
]
for key in required_true:
    if invariants.get(key) is not True:
        raise SystemExit(f"LR1_INVARIANT_DISABLED:{key}")

collector = (root / "integration/llama.cpp/shim/slha_rank_dataset.cpp").read_text(encoding="utf-8")
for needle in ["int chunk = 0;", "if (s.on) ++s.chunk;"]:
    if needle not in collector:
        raise SystemExit(f"LR1_COLLECTOR_INDEXING_GUARD_MISSING:{needle}")

shim = (root / "integration/llama.cpp/shim/slha_llama.cpp").read_text(encoding="utf-8")
for needle in [
    "void slha_k_clear_all()",
    "slha_rank_dataset::add_keys",
    "slha_rank_dataset::begin_chunk();",
]:
    if needle not in shim:
        raise SystemExit(f"LR1_CLEAR_LIFECYCLE_GUARD_MISSING:{needle}")

patch = (root / "integration/llama.cpp/patches/0001-slha-k-passthrough.patch").read_text(encoding="utf-8")
if "llama_memory_clear" not in patch or "slha_k_clear_all();" not in patch:
    raise SystemExit("LR1_LLAMA_CLEAR_HOOK_GUARD_MISSING")

lib = (root / "scirust/src/lib.rs").read_text(encoding="utf-8")
if "pub mod lr1_contract;" not in lib:
    raise SystemExit("LR1_RUNTIME_CONTRACT_MODULE_NOT_EXPOSED")

runtime_contract = (root / "scirust/src/lr1_contract.rs").read_text(encoding="utf-8")
for needle in [
    'pub const CANDIDATE_ID: &str = "slha-lr1-pairwise-top16-all-layers-v1";',
    'pub const MODEL_SHA256: &str = "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6";',
    'pub const LLAMA_COMMIT: &str = "fdb1db877c526ec90f668eca1b858da5dba85560";',
    'pub const TRAINING_CHUNKS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];',
    'pub const VALIDATION_CHUNKS: [usize; 4] = [13, 14, 15, 16];',
    'pub fn validate_file(path: &str) -> Result<ValidatedContract, String>',
    'exact_string(stage_a, "ranking_metric_row_rule", "n_visible > top_k")',
    'exact_bool(stage_b, "enabled_by_this_contract", false)',
]:
    if needle not in runtime_contract:
        raise SystemExit(f"LR1_RUNTIME_CONTRACT_GUARD_MISSING:{needle}")

reader = (root / "scirust/src/rank_dataset.rs").read_text(encoding="utf-8")
if "validate_chunk_layout" not in reader:
    raise SystemExit("LR1_DATASET_LAYOUT_VALIDATOR_MISSING")

trainer = (root / "scirust/src/bin/slha_train_lr1.rs").read_text(encoding="utf-8")
for needle in [
    "const TOP_K: usize = 16;",
    "const STORAGE_SLOTS: usize = 17;",
    "const POPULATED_CHUNKS: [usize; 16]",
    "const TRAINING_CHUNKS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];",
    "Objective::PairwiseTopK",
    "validate_chunk_layout(STORAGE_SLOTS, &POPULATED_CHUNKS)",
    "lr1_contract::validate_file(&args.contract)",
    '\"contract_semantically_validated\":true',
    '\"short_context_policy\":\"retain_existing_objective_semantics\"',
]:
    if needle not in trainer:
        raise SystemExit(f"LR1_FROZEN_TRAINER_GUARD_MISSING:{needle}")
for forbidden in ["--top-k", "--layers", "--objective", "--learning-rate", "--epochs"]:
    if f'"{forbidden}"' in trainer:
        raise SystemExit(f"LR1_FROZEN_TRAINER_EXPOSES_TUNING:{forbidden}")
if "n_visible < TOP_K" in trainer or "n_visible <= TOP_K" in trainer:
    raise SystemExit("LR1_TRAINER_FILTERS_SHORT_CONTEXT_ROWS")
trainer_main = trainer.index("fn main()")
trainer_validate = trainer.index("lr1_contract::validate_file(&args.contract)", trainer_main)
trainer_optimizer = trainer.index("train_ranking(&rows", trainer_main)
if not trainer_main < trainer_validate < trainer_optimizer:
    raise SystemExit("LR1_TRAINER_CONTRACT_VALIDATION_ORDER_INVALID")

evaluator = (root / "scirust/src/bin/slha_eval_rank.rs").read_text(encoding="utf-8")
for needle in [
    "const EXPECTED_TOP_K: usize = 16;",
    "const EXPECTED_STORAGE_SLOTS: usize = 17;",
    "const EXPECTED_POPULATED: [usize; 16]",
    "const EXPECTED_SPLIT: [usize; 4] = [13, 14, 15, 16];",
    "validate_chunk_layout(EXPECTED_STORAGE_SLOTS, &EXPECTED_POPULATED)",
    "LatentCodec::Mixed",
    "lr1_contract::validate_file(&args.contract)",
    '\"semantically_validated\":true',
    "ranking_rows: u64",
    "if baseline.len() > k",
    '\"ranking_rows\"',
    '\"spearman_rows\"',
    '\"geometry_rows\"',
    '\"ranking_row_rule\":\"n_visible > top_k\"',
]:
    if needle not in evaluator:
        raise SystemExit(f"LR1_FROZEN_EVALUATOR_GUARD_MISSING:{needle}")
evaluator_main = evaluator.index("fn main()")
evaluator_validate = evaluator.index("lr1_contract::validate_file(&args.contract)", evaluator_main)
evaluator_dataset_read = evaluator.index("let rows = read_layer", evaluator_main)
if not evaluator_main < evaluator_validate < evaluator_dataset_read:
    raise SystemExit("LR1_EVALUATOR_CONTRACT_VALIDATION_ORDER_INVALID")

external_k = (root / "integration/llama.cpp/shim/slha_external_k.cpp").read_text(encoding="utf-8")
for needle in [
    "external K forbids SLHA_ORACLE_METRICS_JSON because paired baseline logits are absent",
    "external K forbids SLHA_SCALE_FIT_JSON because fitting consumes paired baseline logits",
    "external K forbids SLHA_RANK_DATASET_DIR because ranking labels require baseline logits",
]:
    if needle not in external_k:
        raise SystemExit(f"LR1_EXTERNAL_K_FAIL_CLOSED_GUARD_MISSING:{needle}")

compact = {
    "candidate_id": c["candidate_id"],
    "objective": c["candidate"]["objective"],
    "top_k": c["candidate"]["top_k"],
    "layers": c["candidate"]["layers"],
    "storage_slots": source["rank_dataset_storage_slots"],
    "training_chunks": train,
    "validation_chunks": valid,
    "ranking_metric_row_rule": stage_a["ranking_metric_row_rule"],
    "runtime_contract_validation": True,
    "diagnostic_sha256": rank_diag["sha256"],
    "protected_holdout_sha256": holdout["sha256"],
    "stage_b_enabled": c["stage_b"]["enabled_by_this_contract"],
}
print("SLHA_LR1_CONTRACT_VALID=" + json.dumps(compact, sort_keys=True, separators=(",", ":")))
PY
