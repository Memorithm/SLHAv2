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

train = source.get("training_chunks")
valid = source.get("validation_chunks")
if train != list(range(12)) or valid != list(range(12, 16)):
    raise SystemExit("LR1_SPLIT_DRIFT")
if set(train) & set(valid):
    raise SystemExit("LR1_SPLIT_OVERLAP")
if sorted(train + valid) != list(range(source.get("rank_dataset_chunks", -1))):
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
if (c.get("stage_a") or {}).get("protected_holdout_allowed") is not False:
    raise SystemExit("LR1_STAGE_A_HOLDOUT_ENABLED")
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

trainer = (root / "scirust/src/bin/slha_train_rank.rs").read_text(encoding="utf-8")
for needle in [
    '"pairwise-topk"',
    'const MAX_TOPK: usize = 64;',
    '"--top-k"',
    '"--layers"',
    '"--split-chunks"',
    '"--execution-binding"',
    '"--normalisation-manifest"',
]:
    if needle not in trainer:
        raise SystemExit(f"LR1_TRAINER_CAPABILITY_MISSING:{needle}")

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
    "training_chunks": train,
    "validation_chunks": valid,
    "diagnostic_sha256": rank_diag["sha256"],
    "protected_holdout_sha256": holdout["sha256"],
    "stage_b_enabled": c["stage_b"]["enabled_by_this_contract"],
}
print("SLHA_LR1_CONTRACT_VALID=" + json.dumps(compact, sort_keys=True, separators=(",", ":")))
PY
