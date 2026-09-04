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
compgen -G "$WEIGHTS_DIR/layer-*.slhw" >/dev/null || {
    echo "ERROR: no layer-*.slhw files in $WEIGHTS_DIR" >&2
    exit 2
}
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

CORPUS_SHA256="$(sha256sum "$CORPUS" | awk '{print $1}')"
HOLDOUT_SHA256="$(sha256sum "$HOLDOUT" | awk '{print $1}')"
MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
WEIGHTS_MANIFEST_SHA256="$(sha256sum "$WEIGHTS_DIR/manifest.json" | awk '{print $1}')"
if [[ "$CORPUS_SHA256" == "$HOLDOUT_SHA256" ]]; then
    echo "ERROR: rank intervention corpus must be disjoint from the protected perplexity holdout" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
REPORT_JSON="$OUTPUT_DIR/rank-intervention-diagnostic.json"
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

run_slha() {
    local label="$1"
    local oracle="$2"
    local log="$OUTPUT_DIR/$label.log"
    rm -f "$log"
    local oracle_env=()
    if [[ -n "$oracle" ]]; then
        oracle_env+=("SLHA_SCORE_ORACLE=$oracle")
    fi
    set +e
    env \
        -u SLHA_EXTERNAL_K \
        -u SLHA_CCOS \
        -u SLHA_CCOS_BUDGET_BYTES \
        -u SLHA_CCOS_IMPORTANCE_TEMPERATURE \
        -u SLHA_ORACLE_METRICS_JSON \
        -u SLHA_SCALE_FIT_JSON \
        -u SLHA_RANK_DATASET_DIR \
        SLHA_KV_MODE=tilestore \
        SLHA_SCORE_MODE=replace \
        SLHA_SCORE_LAYERS=all \
        SLHA_WEIGHTS_DIR="$WEIGHTS_DIR" \
        SLHA_CODEC="$CODEC" \
        "${oracle_env[@]}" \
        LC_ALL=C "$PPL_BIN" "${PPL_ARGS[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    [[ "$rc" -eq 0 ]] || { echo "ERROR: $label failed with exit code $rc" >&2; exit "$rc"; }
}

# Fixed diagnostic matrix. This is not an adaptive sweep and no case is selected
# or removed based on observed PPL.
run_baseline baseline
run_slha strict ""
run_slha baseline_identity baseline-identity
run_slha slha_identity slha-identity
run_slha rankA_baseline_rank_slha_values baseline-rank-slha-values
run_slha rankB_slha_rank_baseline_values slha-rank-baseline-values
run_slha baseline_topk1 baseline-topk:1
run_slha baseline_topk16 baseline-topk:16

python3 - \
    "$OUTPUT_DIR" "$REPORT_JSON" "$CORPUS" "$CORPUS_SHA256" "$HOLDOUT_SHA256" \
    "$MODEL_SHA256" "$WEIGHTS_MANIFEST_SHA256" "$SLHA_COMMIT" "$LLAMA_EXPECTED" \
    "$CTX_SIZE" "$CHUNKS" "$THREADS" "$GPU_LAYERS" "$CODEC" <<'PY'
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
    "strict": None,
    "baseline_identity": "baseline-identity",
    "slha_identity": "slha-identity",
    "rankA_baseline_rank_slha_values": "baseline-rank-slha-values",
    "rankB_slha_rank_baseline_values": "slha-rank-baseline-values",
    "baseline_topk1": "baseline-topk:1",
    "baseline_topk16": "baseline-topk:16",
}

ppl_re = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9eE+.-]+)")

def parse_oracle_summary(text):
    blocks = re.findall(
        r"SLHA_SCORE_ORACLE_SUMMARY[^\n]*\n(.*?)END_SLHA_SCORE_ORACLE_SUMMARY",
        text,
        flags=re.DOTALL,
    )
    if not blocks:
        return None
    out = {}
    for line in blocks[-1].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out

parsed = {}
for label, expected_oracle in cases.items():
    path = os.path.join(output_dir, label + ".log")
    text = open(path, encoding="utf-8", errors="replace").read()
    matches = ppl_re.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"{label}: expected exactly one final PPL, got {len(matches)}")
    ppl = float(matches[0])
    if not math.isfinite(ppl):
        raise SystemExit(f"{label}: non-finite PPL")
    summary = parse_oracle_summary(text)
    if expected_oracle is not None:
        if summary is None:
            raise SystemExit(f"{label}: missing score-oracle summary")
        if summary.get("oracle_mode_valid") != "true":
            raise SystemExit(f"{label}: oracle mode invalid: {summary}")
        if summary.get("canonical") != expected_oracle:
            raise SystemExit(f"{label}: canonical oracle mismatch: {summary}")
        for key in (
            "oracle_nonfinite_input",
            "oracle_invalid_permutation",
            "oracle_partial_write",
            "oracle_invariant_failed",
        ):
            if int(summary.get(key, "-1")) != 0:
                raise SystemExit(f"{label}: {key} is nonzero: {summary}")
        if int(summary.get("oracle_vectors", "0")) <= 0:
            raise SystemExit(f"{label}: no oracle vectors executed")
    parsed[label] = {
        "ppl": ppl,
        "oracle": expected_oracle,
        "oracle_summary": summary,
        "log_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
    }

baseline = parsed["baseline"]["ppl"]
strict = parsed["strict"]["ppl"]
bident = parsed["baseline_identity"]["ppl"]
sident = parsed["slha_identity"]["ppl"]
rank_a = parsed["rankA_baseline_rank_slha_values"]["ppl"]
rank_b = parsed["rankB_slha_rank_baseline_values"]["ppl"]
top1 = parsed["baseline_topk1"]["ppl"]
top16 = parsed["baseline_topk16"]["ppl"]

# Identity controls are required for causal interpretation. Comparison uses the
# exact decimal value printed by llama-perplexity and parsed above; no tolerance
# is introduced after seeing the result.
if bident != baseline:
    raise SystemExit(f"baseline identity control mismatch: baseline={baseline}, identity={bident}")
if sident != strict:
    raise SystemExit(f"SLHA identity control mismatch: strict={strict}, identity={sident}")

gap = strict - baseline

def recovery(ppl):
    amount = strict - ppl
    fraction = amount / gap if gap != 0.0 else None
    return {"amount": amount, "fraction_of_strict_minus_baseline_gap": fraction}

report = {
    "schema": "slha_rank_intervention_diagnostic_v1",
    "status": "DIAGNOSTIC_ONLY_NOT_QUALITY_PROMOTION",
    "question": "On the TinyStories diagnostic lineage, how much of the strict replacement PPL gap changes when baseline ranking or baseline score-value geometry is transplanted into the paired SLHA row?",
    "execution_mode": "historical_tilestore_replace_with_paired_baseline_qk",
    "external_k": False,
    "ccos": False,
    "paired_baseline_logits": True,
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
        "context_size": int(context_size),
        "chunks": int(chunks),
        "threads": int(threads),
        "gpu_layers": int(gpu_layers),
        "codec": codec,
    },
    "identity_controls": {
        "baseline_equals_baseline_identity": True,
        "strict_equals_slha_identity": True,
    },
    "endpoints": {
        "baseline_ppl": baseline,
        "strict_replacement_ppl": strict,
        "strict_minus_baseline_gap": gap,
    },
    "interventions": {
        "baseline_rank_slha_values": {
            "ppl": rank_a,
            "recovery": recovery(rank_a),
            "order_preserving_residual_ppl_above_baseline": rank_a - baseline,
        },
        "slha_rank_baseline_values": {
            "ppl": rank_b,
            "change_from_strict": strict - rank_b,
            "recovery": recovery(rank_b),
        },
        "baseline_topk1": {
            "ppl": top1,
            "recovery": recovery(top1),
        },
        "baseline_topk16": {
            "ppl": top16,
            "recovery": recovery(top16),
        },
    },
    "cases": parsed,
    "limitations": [
        "Every intervention consumes exact paired baseline Q*K and is therefore diagnostic, not deployable.",
        "This corpus is separate from the protected quality holdout and cannot promote end-to-end quality claims.",
        "Recovery fractions are descriptive contrasts, not an additive variance decomposition.",
        "Ranking and score geometry may interact; the two rank/value transplants are not complementary partitions.",
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
    "baseline_rank_slha_values_ppl": rank_a,
    "baseline_rank_recovery_fraction": report["interventions"]["baseline_rank_slha_values"]["recovery"]["fraction_of_strict_minus_baseline_gap"],
    "order_preserving_residual_ppl_above_baseline": rank_a - baseline,
    "slha_rank_baseline_values_ppl": rank_b,
    "top1_ppl": top1,
    "top1_recovery_fraction": report["interventions"]["baseline_topk1"]["recovery"]["fraction_of_strict_minus_baseline_gap"],
    "top16_ppl": top16,
    "top16_recovery_fraction": report["interventions"]["baseline_topk16"]["recovery"]["fraction_of_strict_minus_baseline_gap"],
}
print("SLHA_RANK_INTERVENTION_EVIDENCE=" + json.dumps(compact, sort_keys=True, separators=(",", ":")))
PY

python3 -m json.tool "$REPORT_JSON" >/dev/null
printf 'SLHA rank intervention diagnostic report: %s\n' "$REPORT_JSON"