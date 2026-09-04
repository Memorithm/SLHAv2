#!/usr/bin/env bash
set -euo pipefail

MODEL=""
WEIGHTS_DIR=""
CORPUS=""
OUTPUT_DIR=""
PAIR_WORK=""
CTX_SIZE=128
CHUNKS=1
THREADS=2
GPU_LAYERS=0
CODEC="mixed"
CACHE_TYPE_K="f16"
CACHE_TYPE_V="f16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:?missing value for --model}"; shift 2 ;;
        --weights-dir) WEIGHTS_DIR="${2:?missing value for --weights-dir}"; shift 2 ;;
        --corpus) CORPUS="${2:?missing value for --corpus}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?missing value for --output-dir}"; shift 2 ;;
        --pair-work) PAIR_WORK="${2:?missing value for --pair-work}"; shift 2 ;;
        --context-size) CTX_SIZE="${2:?missing value for --context-size}"; shift 2 ;;
        --chunks) CHUNKS="${2:?missing value for --chunks}"; shift 2 ;;
        --threads) THREADS="${2:?missing value for --threads}"; shift 2 ;;
        --gpu-layers) GPU_LAYERS="${2:?missing value for --gpu-layers}"; shift 2 ;;
        --codec) CODEC="${2:?missing value for --codec}"; shift 2 ;;
        --cache-type-k) CACHE_TYPE_K="${2:?missing value for --cache-type-k}"; shift 2 ;;
        --cache-type-v) CACHE_TYPE_V="${2:?missing value for --cache-type-v}"; shift 2 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

for name in MODEL WEIGHTS_DIR CORPUS OUTPUT_DIR PAIR_WORK; do
    [[ -n "${!name}" ]] || { echo "ERROR: --${name,,} is required" >&2; exit 2; }
done
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }
[[ -d "$WEIGHTS_DIR" ]] || { echo "ERROR: weights directory not found: $WEIGHTS_DIR" >&2; exit 2; }
[[ -f "$WEIGHTS_DIR/manifest.json" ]] || { echo "ERROR: missing weights manifest" >&2; exit 2; }
[[ -s "$CORPUS" ]] || { echo "ERROR: diagnostic corpus is missing or empty: $CORPUS" >&2; exit 2; }
for value in "$CTX_SIZE" "$CHUNKS" "$THREADS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: context/chunks/threads must be positive integers" >&2; exit 2; }
done
[[ "$GPU_LAYERS" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpu-layers must be a non-negative integer" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SLHA_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
LLAMA_EXPECTED="fdb1db877c526ec90f668eca1b858da5dba85560"
LLAMA_DIR="$PAIR_WORK/llama.cpp-b9860"
PPL_BIN="$LLAMA_DIR/build/bin/llama-perplexity"
HOLDOUT="$REPO_ROOT/integration/llama.cpp/fixtures/tinystories_synthetic_holdout.txt"

[[ -x "$PPL_BIN" ]] || {
    echo "ERROR: missing patched llama-perplexity at $PPL_BIN; run run_real_pair.sh with --work $PAIR_WORK first" >&2
    exit 2
}
[[ -f "$HOLDOUT" ]] || { echo "ERROR: protected holdout fixture is missing" >&2; exit 2; }
[[ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" == "$LLAMA_EXPECTED" ]] || {
    echo "ERROR: pair-work llama.cpp commit is not the pinned b9860 commit" >&2
    exit 2
}

NUM_LAYERS="$(python3 - "$WEIGHTS_DIR/manifest.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["num_layers"]))
PY
)"
[[ "$NUM_LAYERS" -eq 6 ]] || {
    echo "ERROR: this fixed TinyStories localization matrix requires exactly 6 layers; got $NUM_LAYERS" >&2
    exit 2
}

CORPUS_SHA256="$(sha256sum "$CORPUS" | awk '{print $1}')"
HOLDOUT_SHA256="$(sha256sum "$HOLDOUT" | awk '{print $1}')"
MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
WEIGHTS_MANIFEST_SHA256="$(sha256sum "$WEIGHTS_DIR/manifest.json" | awk '{print $1}')"
if [[ "$CORPUS_SHA256" == "$HOLDOUT_SHA256" ]]; then
    echo "ERROR: localization corpus must be disjoint from the protected quality holdout" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
REPORT_JSON="$OUTPUT_DIR/layer-localization-diagnostic.json"
rm -f "$REPORT_JSON"

PPL_ARGS=(
    -m "$MODEL"
    -f "$CORPUS"
    --chunks "$CHUNKS"
    -c "$CTX_SIZE"
    --batch-size "$CTX_SIZE"
    -t "$THREADS"
    --parallel 1
    --flash-attn off
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --gpu-layers "$GPU_LAYERS"
)

run_baseline() {
    local label="$1"
    local log="$OUTPUT_DIR/$label.log"
    rm -f "$log"
    set +e
    env \
        -u SLHA_KV_MODE \
        -u SLHA_SCORE_MODE \
        -u SLHA_SCORE_LAYERS \
        -u SLHA_WEIGHTS_DIR \
        -u SLHA_CODEC \
        -u SLHA_SCORE_ORACLE \
        -u SLHA_EXTERNAL_K \
        -u SLHA_CCOS \
        -u SLHA_ORACLE_METRICS_JSON \
        -u SLHA_SCALE_FIT_JSON \
        -u SLHA_RANK_DATASET_DIR \
        LC_ALL=C "$PPL_BIN" "${PPL_ARGS[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    [[ "$rc" -eq 0 ]] || { echo "ERROR: $label failed with exit code $rc" >&2; exit "$rc"; }
}

run_masked() {
    local label="$1"
    local mask="$2"
    local log="$OUTPUT_DIR/$label.log"
    rm -f "$log"
    set +e
    env \
        -u SLHA_SCORE_ORACLE \
        -u SLHA_EXTERNAL_K \
        -u SLHA_CCOS \
        -u SLHA_CCOS_BUDGET_BYTES \
        -u SLHA_CCOS_IMPORTANCE_TEMPERATURE \
        -u SLHA_ORACLE_METRICS_JSON \
        -u SLHA_SCALE_FIT_JSON \
        -u SLHA_RANK_DATASET_DIR \
        SLHA_KV_MODE=tilestore \
        SLHA_SCORE_MODE=replace \
        SLHA_SCORE_LAYERS="$mask" \
        SLHA_WEIGHTS_DIR="$WEIGHTS_DIR" \
        SLHA_CODEC="$CODEC" \
        LC_ALL=C "$PPL_BIN" "${PPL_ARGS[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    [[ "$rc" -eq 0 ]] || { echo "ERROR: $label failed with exit code $rc" >&2; exit "$rc"; }
}

# Fixed, predeclared 6-layer localization matrix. Unselected layers retain exact
# baseline Q*K; selected layers use strict SLHA replacement. No case is added,
# removed, selected, or retried based on observed PPL.
run_baseline baseline
run_masked none none
run_masked strict all
run_masked strict_repeat all

run_masked only_0 0
run_masked only_1 1
run_masked only_2 2
run_masked only_3 3
run_masked only_4 4
run_masked only_5 5

run_masked rescue_0 1-5
run_masked rescue_1 0,2-5
run_masked rescue_2 0-1,3-5
run_masked rescue_3 0-2,4-5
run_masked rescue_4 0-3,5
run_masked rescue_5 0-4

run_masked rescue_early_0_1 2-5
run_masked rescue_mid_2_3 0-1,4-5
run_masked rescue_late_4_5 0-3

python3 - \
    "$OUTPUT_DIR" "$REPORT_JSON" "$CORPUS" "$CORPUS_SHA256" "$HOLDOUT_SHA256" \
    "$MODEL_SHA256" "$WEIGHTS_MANIFEST_SHA256" "$SLHA_COMMIT" "$LLAMA_EXPECTED" \
    "$CTX_SIZE" "$CHUNKS" "$THREADS" "$GPU_LAYERS" "$CODEC" <<'PY'
import hashlib
import json
import math
import os
import re
import sys

(
    output_dir,
    report_path,
    corpus_path,
    corpus_sha,
    holdout_sha,
    model_sha,
    weights_manifest_sha,
    slha_commit,
    llama_commit,
    context_size,
    chunks,
    threads,
    gpu_layers,
    codec,
) = sys.argv[1:]

cases = {
    "baseline": None,
    "none": "none",
    "strict": "all",
    "strict_repeat": "all",
    "only_0": "0",
    "only_1": "1",
    "only_2": "2",
    "only_3": "3",
    "only_4": "4",
    "only_5": "5",
    "rescue_0": "1-5",
    "rescue_1": "0,2-5",
    "rescue_2": "0-1,3-5",
    "rescue_3": "0-2,4-5",
    "rescue_4": "0-3,5",
    "rescue_5": "0-4",
    "rescue_early_0_1": "2-5",
    "rescue_mid_2_3": "0-1,4-5",
    "rescue_late_4_5": "0-3",
}
expected_layers = {
    "none": "",
    "strict": "0,1,2,3,4,5",
    "strict_repeat": "0,1,2,3,4,5",
    "only_0": "0",
    "only_1": "1",
    "only_2": "2",
    "only_3": "3",
    "only_4": "4",
    "only_5": "5",
    "rescue_0": "1,2,3,4,5",
    "rescue_1": "0,2,3,4,5",
    "rescue_2": "0,1,3,4,5",
    "rescue_3": "0,1,2,4,5",
    "rescue_4": "0,1,2,3,5",
    "rescue_5": "0,1,2,3,4",
    "rescue_early_0_1": "2,3,4,5",
    "rescue_mid_2_3": "0,1,4,5",
    "rescue_late_4_5": "0,1,2,3",
}

ppl_re = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9eE+.-]+)")

def parse_summary(text, begin, end):
    blocks = re.findall(re.escape(begin) + r"[^\n]*\n(.*?)" + re.escape(end), text, flags=re.DOTALL)
    if not blocks:
        return None
    out = {}
    for line in blocks[-1].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out

parsed = {}
for label, mask in cases.items():
    path = os.path.join(output_dir, label + ".log")
    text = open(path, encoding="utf-8", errors="replace").read()
    matches = ppl_re.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"{label}: expected exactly one final PPL, got {len(matches)}")
    ppl = float(matches[0])
    if not math.isfinite(ppl):
        raise SystemExit(f"{label}: non-finite PPL")

    mask_summary = None
    replace_summary = None
    if mask is not None:
        mask_summary = parse_summary(text, "SLHA_SCORE_MASK_SUMMARY", "END_SLHA_SCORE_MASK_SUMMARY")
        if mask_summary is None:
            raise SystemExit(f"{label}: missing score-mask summary")
        if mask_summary.get("requested_spec") != mask:
            raise SystemExit(f"{label}: requested mask mismatch: {mask_summary}")
        if mask_summary.get("mask_valid") != "true" or mask_summary.get("mask_error") != "false":
            raise SystemExit(f"{label}: invalid score mask: {mask_summary}")
        if mask_summary.get("resolved_layers", "") != expected_layers[label]:
            raise SystemExit(f"{label}: resolved layer mismatch: {mask_summary}")
        if mask_summary.get("executed_layers", "") != expected_layers[label]:
            raise SystemExit(f"{label}: executed layer mismatch: {mask_summary}")

        replace_summary = parse_summary(text, "SLHA_REPLACE_SUMMARY", "END_SLHA_REPLACE_SUMMARY")
        if replace_summary is None:
            # Historical summary has no explicit END marker; fall back to the
            # block ending at the external-store summary when present.
            blocks = re.findall(r"SLHA_REPLACE_SUMMARY\n(.*?)(?:SLHA_EXTERNAL_K_STORE|\Z)", text, flags=re.DOTALL)
            if blocks:
                replace_summary = {}
                for line in blocks[-1].splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        replace_summary[key.strip()] = value.strip()
        if mask != "none":
            if replace_summary is None or replace_summary.get("valid") != "true":
                raise SystemExit(f"{label}: replacement summary missing or invalid: {replace_summary}")
            if int(replace_summary.get("failed_vectors", "-1")) != 0:
                raise SystemExit(f"{label}: replacement failures: {replace_summary}")

    parsed[label] = {
        "ppl": ppl,
        "mask": mask,
        "mask_summary": mask_summary,
        "log_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }

baseline = parsed["baseline"]["ppl"]
none = parsed["none"]["ppl"]
strict = parsed["strict"]["ppl"]
strict_repeat = parsed["strict_repeat"]["ppl"]
if none != baseline:
    raise SystemExit(f"none-mask identity mismatch: baseline={baseline}, none={none}")
if strict_repeat != strict:
    raise SystemExit(f"strict repeat mismatch: strict={strict}, repeat={strict_repeat}")

gap = strict - baseline

def rescue(label):
    ppl = parsed[label]["ppl"]
    amount = strict - ppl
    return {
        "ppl": ppl,
        "change_from_strict": amount,
        "fraction_of_strict_minus_baseline_gap": amount / gap if gap != 0.0 else None,
    }

def only(label):
    ppl = parsed[label]["ppl"]
    return {
        "ppl": ppl,
        "delta_from_baseline": ppl - baseline,
    }

report = {
    "schema": "slha_layer_localization_diagnostic_v1",
    "status": "DIAGNOSTIC_ONLY_NOT_QUALITY_PROMOTION",
    "question": "Within one paired TinyStories diagnostic run, which strict SLHA replacement layers or layer groups account for the observed PPL gap when all unselected layers retain exact baseline Q*K?",
    "execution_mode": "historical_tilestore_replace_with_layer_masked_baseline_qk_passthrough",
    "external_k": False,
    "ccos": False,
    "score_oracle": False,
    "fixed_matrix": list(cases),
    "corpus": {
        "path": os.path.relpath(os.path.abspath(corpus_path), os.getcwd()),
        "sha256": corpus_sha,
        "protected_holdout_sha256": holdout_sha,
        "is_protected_holdout": False,
    },
    "model_sha256": model_sha,
    "weights_manifest_sha256": weights_manifest_sha,
    "slhav2_commit": slha_commit,
    "llama_cpp_commit": llama_commit,
    "configuration": {
        "num_layers": 6,
        "context_size": int(context_size),
        "chunks": int(chunks),
        "threads": int(threads),
        "gpu_layers": int(gpu_layers),
        "codec": codec,
    },
    "identity_controls": {
        "ordinary_baseline_equals_none_mask": True,
        "strict_all_equals_strict_repeat": True,
    },
    "endpoints": {
        "baseline_ppl": baseline,
        "strict_all_ppl": strict,
        "strict_minus_baseline_gap": gap,
    },
    "single_layer_replacement": {str(i): only(f"only_{i}") for i in range(6)},
    "leave_one_layer_baseline_rescue": {str(i): rescue(f"rescue_{i}") for i in range(6)},
    "two_layer_baseline_rescue": {
        "early_0_1": rescue("rescue_early_0_1"),
        "mid_2_3": rescue("rescue_mid_2_3"),
        "late_4_5": rescue("rescue_late_4_5"),
    },
    "cases": parsed,
    "limitations": [
        "This is a fixed non-protected diagnostic intervention matrix, not a deployable scorer or cache policy.",
        "Unselected layers pass exact baseline Q*K and therefore represent diagnostic rescues unavailable to compressed-only deployment.",
        "Layer and group effects can interact; single-layer deltas and leave-one-out rescues are not additive decompositions.",
        "Absolute PPL values are interpreted only within this paired run; historical runs are not used as controls.",
        "This run does not measure physical external-K memory behavior, CCOS residency, or deployment performance.",
    ],
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")

compact = {
    "schema": report["schema"],
    "status": report["status"],
    "corpus_sha256": corpus_sha,
    "baseline_ppl": baseline,
    "strict_ppl": strict,
    "gap": gap,
    "single_layer_delta_from_baseline": {
        k: v["delta_from_baseline"] for k, v in report["single_layer_replacement"].items()
    },
    "leave_one_out_recovery_fraction": {
        k: v["fraction_of_strict_minus_baseline_gap"]
        for k, v in report["leave_one_layer_baseline_rescue"].items()
    },
    "two_layer_recovery_fraction": {
        k: v["fraction_of_strict_minus_baseline_gap"]
        for k, v in report["two_layer_baseline_rescue"].items()
    },
}
print("SLHA_LAYER_LOCALIZATION_EVIDENCE=" + json.dumps(compact, sort_keys=True, separators=(",", ":")))
PY

python3 -m json.tool "$REPORT_JSON" >/dev/null
printf 'SLHA layer localization diagnostic report: %s\n' "$REPORT_JSON"
