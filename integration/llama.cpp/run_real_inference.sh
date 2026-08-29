#!/usr/bin/env bash
# Reproducible real-model baseline / physical-SLHA generation runner.
#
# This script intentionally uses the SAME patched llama.cpp binary for both
# modes. `baseline` leaves SLHA disabled; `external` enables the physical K
# path. It records only observed process/engine values and never estimates
# whole-process savings from the 128-byte tile format.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "external" ]]; then
    echo "usage: $0 baseline|external --model MODEL.gguf --prompt TEXT --output-json REPORT.json [options]" >&2
    exit 2
fi
shift

MODEL=""
PROMPT=""
OUTPUT_JSON=""
WEIGHTS_DIR=""
WORK="${WORK:-/tmp/slha-real-inference}"
MAX_TOKENS=64
CTX_SIZE=2048
THREADS=4
SEED=1
GPU_LAYERS=0
CODEC="mixed"
CACHE_TYPE_K="f16"
CACHE_TYPE_V="f16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:?missing value for --model}"; shift 2 ;;
        --prompt) PROMPT="${2:?missing value for --prompt}"; shift 2 ;;
        --output-json) OUTPUT_JSON="${2:?missing value for --output-json}"; shift 2 ;;
        --weights-dir) WEIGHTS_DIR="${2:?missing value for --weights-dir}"; shift 2 ;;
        --work) WORK="${2:?missing value for --work}"; shift 2 ;;
        --max-tokens) MAX_TOKENS="${2:?missing value for --max-tokens}"; shift 2 ;;
        --context-size) CTX_SIZE="${2:?missing value for --context-size}"; shift 2 ;;
        --threads) THREADS="${2:?missing value for --threads}"; shift 2 ;;
        --seed) SEED="${2:?missing value for --seed}"; shift 2 ;;
        --gpu-layers) GPU_LAYERS="${2:?missing value for --gpu-layers}"; shift 2 ;;
        --codec) CODEC="${2:?missing value for --codec}"; shift 2 ;;
        --cache-type-k) CACHE_TYPE_K="${2:?missing value for --cache-type-k}"; shift 2 ;;
        --cache-type-v) CACHE_TYPE_V="${2:?missing value for --cache-type-v}"; shift 2 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "ERROR: --model is required" >&2; exit 2; }
[[ -n "$PROMPT" ]] || { echo "ERROR: --prompt is required" >&2; exit 2; }
[[ -n "$OUTPUT_JSON" ]] || { echo "ERROR: --output-json is required" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }
command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 2; }
command -v cmake >/dev/null || { echo "ERROR: cmake is required" >&2; exit 2; }
command -v cargo >/dev/null || { echo "ERROR: cargo is required" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }
[[ -x /usr/bin/time ]] || { echo "ERROR: /usr/bin/time is required (Debian/Ubuntu package: time)" >&2; exit 2; }

if [[ "$MODE" == "external" ]]; then
    [[ -n "$WEIGHTS_DIR" ]] || { echo "ERROR: external mode requires --weights-dir" >&2; exit 2; }
    [[ -d "$WEIGHTS_DIR" ]] || { echo "ERROR: weights directory not found: $WEIGHTS_DIR" >&2; exit 2; }
    [[ -f "$WEIGHTS_DIR/manifest.json" ]] || { echo "ERROR: missing $WEIGHTS_DIR/manifest.json" >&2; exit 2; }
    compgen -G "$WEIGHTS_DIR/layer-*.slhw" >/dev/null || {
        echo "ERROR: no layer-*.slhw projection weights in $WEIGHTS_DIR" >&2
        exit 2
    }
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
REPORTER="$SCRIPT_DIR/scripts/report_real_inference.py"
[[ -f "$REPORTER" ]] || { echo "ERROR: missing report parser: $REPORTER" >&2; exit 2; }

SLHA_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
LLAMA_TAG="b9860"
LLAMA_EXPECTED="fdb1db877c526ec90f668eca1b858da5dba85560"
LLAMA_DIR="$WORK/llama.cpp-$LLAMA_TAG"
RUN_DIR="$WORK/runs/$(date -u +%Y%m%dT%H%M%SZ)-$MODE"
LOG_FILE="$RUN_DIR/inference.log"
TIME_FILE="$RUN_DIR/time.txt"
mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_JSON")"

printf '== SLHA real inference ==\n'
printf 'mode            : %s\n' "$MODE"
printf 'SLHAv2 commit   : %s\n' "$SLHA_COMMIT"
printf 'llama.cpp       : %s (%s)\n' "$LLAMA_TAG" "$LLAMA_EXPECTED"
printf 'model           : %s\n' "$MODEL"
printf 'context         : %s\n' "$CTX_SIZE"
printf 'max tokens      : %s\n' "$MAX_TOKENS"
printf 'threads         : %s\n' "$THREADS"
printf 'GPU layers      : %s\n' "$GPU_LAYERS"

( cd "$REPO_ROOT" && cargo --locked build --release -p slha-c )

# The checkout under WORK is disposable by design. Reset it to the exact pinned
# engine revision on every run so baseline and external cannot accidentally use
# different source trees.
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    rm -rf "$LLAMA_DIR"
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
if [[ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" != "$LLAMA_EXPECTED" ]]; then
    rm -rf "$LLAMA_DIR"
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
test "$(git -C "$LLAMA_DIR" rev-parse HEAD)" = "$LLAMA_EXPECTED"
git -C "$LLAMA_DIR" reset --hard "$LLAMA_EXPECTED" >/dev/null
git -C "$LLAMA_DIR" clean -fdx >/dev/null

normalize_patch() {
    local src="$1"
    local dst="$2"
    awk '
      /^diff --git / { in_hunk = 0 }
      /^@@ / { in_hunk = 1 }
      { if (in_hunk && length($0) == 0) print " "; else print $0 }
    ' "$src" > "$dst"
}

for patch_id in 0001-slha-k-passthrough 0002-slha-external-k 0003-slha-external-k-hardening; do
    src="$REPO_ROOT/integration/llama.cpp/patches/$patch_id.patch"
    normalized="$RUN_DIR/$patch_id.patch"
    [[ -f "$src" ]] || { echo "ERROR: missing patch: $src" >&2; exit 2; }
    normalize_patch "$src" "$normalized"
    git -C "$LLAMA_DIR" apply --recount --check "$normalized"
    git -C "$LLAMA_DIR" apply --recount "$normalized"
done

for shim in \
    slha_llama.hpp slha_llama.cpp \
    slha_external_k.hpp slha_external_k.cpp \
    slha_replace_counters.hpp slha_replace_counters.cpp \
    slha_layer_mask.hpp slha_layer_mask.cpp \
    slha_score_scale.hpp slha_score_scale.cpp \
    slha_scale_fit.hpp slha_scale_fit.cpp \
    slha_score_oracle.hpp slha_score_oracle.cpp \
    slha_oracle_metrics.hpp slha_oracle_metrics.cpp \
    slha_rank_dataset.hpp slha_rank_dataset.cpp; do
    cp "$REPO_ROOT/integration/llama.cpp/shim/$shim" "$LLAMA_DIR/src/$shim"
done

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DSLHA_INTEGRATION=ON \
    -DSLHA_INCLUDE_DIR="$REPO_ROOT/slha-c/include" \
    -DSLHA_LIB="$REPO_ROOT/target/release/libslha.a" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-cli

# Prevent inherited diagnostic knobs from contaminating either control arm.
unset SLHA_EXTERNAL_K SLHA_KV_MODE SLHA_SCORE_MODE SLHA_SCORE_LAYERS
unset SLHA_SCORE_ORACLE SLHA_ORACLE_METRICS_JSON SLHA_SCALE_FIT_JSON
unset SLHA_RANK_DATASET_DIR SLHA_WEIGHTS_DIR SLHA_CODEC

if [[ "$MODE" == "external" ]]; then
    export SLHA_EXTERNAL_K=1
    export SLHA_KV_MODE=tilestore
    export SLHA_SCORE_MODE=replace
    export SLHA_SCORE_LAYERS=all
    export SLHA_WEIGHTS_DIR="$WEIGHTS_DIR"
    export SLHA_CODEC="$CODEC"
fi

CMD=(
    "$LLAMA_DIR/build/bin/llama-cli"
    -m "$MODEL"
    -p "$PROMPT"
    -n "$MAX_TOKENS"
    -c "$CTX_SIZE"
    -t "$THREADS"
    -s "$SEED"
    --temp 0
    --parallel 1
    --batch-size 512
    --flash-attn off
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --gpu-layers "$GPU_LAYERS"
    --perf
)

printf 'command         :'
printf ' %q' "${CMD[@]}"
printf '\n'

set +e
LC_ALL=C /usr/bin/time \
    -f 'max_rss_kb=%M\nelapsed_s=%e\nuser_s=%U\nsys_s=%S' \
    -o "$TIME_FILE" \
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
RUN_RC=${PIPESTATUS[0]}
set -e

MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
MODEL_BYTES="$(stat -c '%s' "$MODEL")"
PROMPT_SHA256="$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"
LOG_SHA256="$(sha256sum "$LOG_FILE" | awk '{print $1}')"

python3 "$REPORTER" \
    --mode "$MODE" \
    --output "$OUTPUT_JSON" \
    --log "$LOG_FILE" \
    --time "$TIME_FILE" \
    --model "$MODEL" \
    --model-sha256 "$MODEL_SHA256" \
    --model-bytes "$MODEL_BYTES" \
    --prompt-sha256 "$PROMPT_SHA256" \
    --slhav2-commit "$SLHA_COMMIT" \
    --llama-commit "$LLAMA_EXPECTED" \
    --context-size "$CTX_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --threads "$THREADS" \
    --seed "$SEED" \
    --gpu-layers "$GPU_LAYERS" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --codec "$CODEC" \
    --exit-code "$RUN_RC" \
    --log-sha256 "$LOG_SHA256" \
    --weights-dir "$WEIGHTS_DIR"

if [[ "$RUN_RC" -ne 0 ]]; then
    echo "ERROR: llama-cli exited with status $RUN_RC; report retained at $OUTPUT_JSON" >&2
    exit "$RUN_RC"
fi

if [[ "$MODE" == "external" ]]; then
    grep -q '^SLHA_EXTERNAL_K_STORE valid=true ' "$LOG_FILE" || {
        echo "ERROR: external run did not report a valid physical tile-store allocation" >&2
        exit 1
    }
    grep -q '^SLHA_REPLACE_SUMMARY$' "$LOG_FILE" || {
        echo "ERROR: external run produced no strict replace summary" >&2
        exit 1
    }
    grep -q '^valid=true$' "$LOG_FILE" || {
        echo "ERROR: external run failed strict replacement coverage" >&2
        exit 1
    }
fi

echo "report: $OUTPUT_JSON"
echo "log   : $LOG_FILE"
