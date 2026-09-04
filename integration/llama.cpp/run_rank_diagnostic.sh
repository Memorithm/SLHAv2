#!/usr/bin/env bash
set -euo pipefail

MODEL=""
WEIGHTS_DIR=""
CORPUS=""
OUTPUT_DIR=""
PAIR_WORK=""
CTX_SIZE=128
CHUNKS=2
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
if [[ "$CORPUS_SHA256" == "$HOLDOUT_SHA256" ]]; then
    echo "ERROR: ranking diagnostic corpus must be disjoint from the protected perplexity holdout" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
RAW_JSON="$OUTPUT_DIR/ranking.raw.json"
REPORT_JSON="$OUTPUT_DIR/ranking-diagnostic.json"
LOG="$OUTPUT_DIR/ranking.log"
rm -f "$RAW_JSON" "$REPORT_JSON" "$LOG"

# This is intentionally the historical paired diagnostic path. Ordinary llama.cpp
# K remains resident so the callback receives real baseline Q*K. Physical external-K
# is forbidden here because that graph intentionally never materialises baseline logits.
unset SLHA_EXTERNAL_K SLHA_CCOS SLHA_CCOS_BUDGET_BYTES SLHA_CCOS_IMPORTANCE_TEMPERATURE
unset SLHA_SCORE_ORACLE SLHA_SCALE_FIT_JSON SLHA_RANK_DATASET_DIR
export SLHA_KV_MODE=tilestore
export SLHA_SCORE_MODE=replace
export SLHA_SCORE_LAYERS=all
export SLHA_WEIGHTS_DIR="$WEIGHTS_DIR"
export SLHA_CODEC="$CODEC"
export SLHA_ORACLE_METRICS_JSON="$RAW_JSON"

set +e
LC_ALL=C "$PPL_BIN" \
    -m "$MODEL" \
    -f "$CORPUS" \
    --chunks "$CHUNKS" \
    -c "$CTX_SIZE" \
    --batch-size "$CTX_SIZE" \
    -t "$THREADS" \
    --parallel 1 \
    --flash-attn off \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --gpu-layers "$GPU_LAYERS" \
    2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -eq 0 ]] || { echo "ERROR: ranking diagnostic failed with exit code $rc" >&2; exit "$rc"; }
[[ -s "$RAW_JSON" ]] || { echo "ERROR: ranking metrics JSON was not produced" >&2; exit 1; }
python3 -m json.tool "$RAW_JSON" >/dev/null

python3 - \
    "$RAW_JSON" "$REPORT_JSON" "$CORPUS" "$CORPUS_SHA256" "$HOLDOUT_SHA256" \
    "$MODEL_SHA256" "$SLHA_COMMIT" "$LLAMA_EXPECTED" "$CTX_SIZE" "$CHUNKS" \
    "$THREADS" "$GPU_LAYERS" "$CODEC" <<'PY'
import json
import os
import sys

(
    raw_path,
    report_path,
    corpus_path,
    corpus_sha,
    holdout_sha,
    model_sha,
    slha_commit,
    llama_commit,
    context_size,
    chunks,
    threads,
    gpu_layers,
    codec,
) = sys.argv[1:]

raw = json.load(open(raw_path, encoding="utf-8"))
if raw.get("schema") != "slha_oracle_active_key_metrics_v1":
    raise SystemExit(f"unexpected ranking metrics schema: {raw.get('schema')!r}")
layers = raw.get("layers") or {}
if not layers:
    raise SystemExit("ranking diagnostic produced no sampled layers")
rows = 0
for layer_id, layer in layers.items():
    layer_rows = int(layer.get("rows", 0))
    if layer_rows <= 0:
        raise SystemExit(f"layer {layer_id} has no sampled ranking rows")
    if int(layer.get("active_accounting_failures", 0)) != 0:
        raise SystemExit(f"layer {layer_id} has active-key accounting failures")
    if layer.get("acct_identity_holds") is not True:
        raise SystemExit(f"layer {layer_id} active-key accounting identity failed")
    rows += layer_rows

report = {
    "schema": "slha_rank_diagnostic_v1",
    "status": "DIAGNOSTIC_ONLY_NOT_QUALITY_PROMOTION",
    "execution_mode": "historical_tilestore_replace_with_paired_baseline_qk",
    "external_k": False,
    "ccos": False,
    "paired_baseline_logits": True,
    "corpus": {
        "path": os.path.relpath(os.path.abspath(corpus_path), os.getcwd()),
        "sha256": corpus_sha,
        "protected_holdout_sha256": holdout_sha,
        "is_protected_holdout": False,
    },
    "model_sha256": model_sha,
    "slhav2_commit": slha_commit,
    "llama_cpp_commit": llama_commit,
    "configuration": {
        "context_size": int(context_size),
        "chunks": int(chunks),
        "threads": int(threads),
        "gpu_layers": int(gpu_layers),
        "codec": codec,
    },
    "sampled_rows": rows,
    "metrics": raw,
    "limitations": [
        "This run keeps the ordinary llama.cpp K cache solely to obtain paired baseline Q*K labels.",
        "It does not measure physical external-K memory behavior or CCOS residency behavior.",
        "Ranking metrics are diagnostic evidence and must not be reported as end-to-end quality proof.",
    ],
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps({"sampled_rows": rows, "layers": len(layers), "corpus_sha256": corpus_sha}, sort_keys=True))
PY

python3 -m json.tool "$REPORT_JSON" >/dev/null
printf 'SLHA ranking diagnostic report: %s\n' "$REPORT_JSON"
