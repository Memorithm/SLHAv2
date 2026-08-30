#!/usr/bin/env bash
# Run an exact paired real-model evaluation through the same patched llama.cpp
# build, once with ordinary KV and once with physical SLHA external-K.
set -euo pipefail

MODEL=""
PROMPT=""
WEIGHTS_DIR=""
OUTPUT_DIR=""
WORK="${WORK:-/tmp/slha-real-pair}"
MAX_TOKENS=64
CTX_SIZE=2048
THREADS=4
GPU_LAYERS=0
CODEC="mixed"
CACHE_TYPE_K="f16"
CACHE_TYPE_V="f16"
CCOS=0
CCOS_BUDGET_BYTES=""
CCOS_IMPORTANCE_TEMPERATURE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:?missing value for --model}"; shift 2 ;;
        --prompt) PROMPT="${2:?missing value for --prompt}"; shift 2 ;;
        --weights-dir) WEIGHTS_DIR="${2:?missing value for --weights-dir}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?missing value for --output-dir}"; shift 2 ;;
        --work) WORK="${2:?missing value for --work}"; shift 2 ;;
        --max-tokens) MAX_TOKENS="${2:?missing value for --max-tokens}"; shift 2 ;;
        --context-size) CTX_SIZE="${2:?missing value for --context-size}"; shift 2 ;;
        --threads) THREADS="${2:?missing value for --threads}"; shift 2 ;;
        --gpu-layers) GPU_LAYERS="${2:?missing value for --gpu-layers}"; shift 2 ;;
        --codec) CODEC="${2:?missing value for --codec}"; shift 2 ;;
        --cache-type-k) CACHE_TYPE_K="${2:?missing value for --cache-type-k}"; shift 2 ;;
        --cache-type-v) CACHE_TYPE_V="${2:?missing value for --cache-type-v}"; shift 2 ;;
        --ccos) CCOS=1; shift ;;
        --ccos-budget-bytes) CCOS_BUDGET_BYTES="${2:?missing value for --ccos-budget-bytes}"; shift 2 ;;
        --ccos-importance-temperature) CCOS_IMPORTANCE_TEMPERATURE="${2:?missing value for --ccos-importance-temperature}"; shift 2 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$MODEL" ]] || { echo "ERROR: --model is required" >&2; exit 2; }
[[ -n "$PROMPT" ]] || { echo "ERROR: --prompt is required" >&2; exit 2; }
[[ -n "$WEIGHTS_DIR" ]] || { echo "ERROR: --weights-dir is required" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "ERROR: --output-dir is required" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }
[[ -d "$WEIGHTS_DIR" ]] || { echo "ERROR: weights directory not found: $WEIGHTS_DIR" >&2; exit 2; }
[[ -f "$WEIGHTS_DIR/manifest.json" ]] || { echo "ERROR: missing weights manifest" >&2; exit 2; }
compgen -G "$WEIGHTS_DIR/layer-*.slhw" >/dev/null || {
    echo "ERROR: no layer-*.slhw files in $WEIGHTS_DIR" >&2
    exit 2
}
if [[ "$CCOS" -ne 1 && ( -n "$CCOS_BUDGET_BYTES" || -n "$CCOS_IMPORTANCE_TEMPERATURE" ) ]]; then
    echo "ERROR: CCOS budget/temperature options require --ccos" >&2
    exit 2
fi
for tool in git cmake cargo python3 sha256sum g++; do
    command -v "$tool" >/dev/null || { echo "ERROR: $tool is required" >&2; exit 2; }
done
[[ -x /usr/bin/time ]] || { echo "ERROR: /usr/bin/time is required (Debian/Ubuntu package: time)" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
SLHA_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
LLAMA_TAG="b9860"
LLAMA_EXPECTED="fdb1db877c526ec90f668eca1b858da5dba85560"
LLAMA_DIR="$WORK/llama.cpp-$LLAMA_TAG"
EVAL_BIN="$WORK/slha-real-eval"
mkdir -p "$WORK" "$OUTPUT_DIR"

MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
PROMPT_SHA256="$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"

printf '== SLHA paired real-model evaluation ==\n'
printf 'SLHAv2 commit : %s\n' "$SLHA_COMMIT"
printf 'llama.cpp     : %s (%s)\n' "$LLAMA_TAG" "$LLAMA_EXPECTED"
printf 'model SHA-256 : %s\n' "$MODEL_SHA256"
printf 'prompt SHA-256: %s\n' "$PROMPT_SHA256"
printf 'context       : %s\n' "$CTX_SIZE"
printf 'max tokens    : %s\n' "$MAX_TOKENS"
printf 'threads       : %s\n' "$THREADS"
printf 'GPU layers    : %s\n' "$GPU_LAYERS"
printf 'external back.: %s\n' "$([[ "$CCOS" -eq 1 ]] && echo ccos_elastic || echo vector)"
printf 'CCOS budget   : %s\n' "${CCOS_BUDGET_BYTES:-default-full-HOT}"
printf 'CCOS temp.    : %s\n' "${CCOS_IMPORTANCE_TEMPERATURE:-default-1.0}"

( cd "$REPO_ROOT" && cargo --locked build --release -p slha-c )

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
    normalized="$WORK/$patch_id.patch"
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
cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama

g++ -O2 -std=c++17 -Wall -Wextra -Werror \
    -I"$LLAMA_DIR/include" \
    -I"$LLAMA_DIR/ggml/include" \
    "$REPO_ROOT/integration/llama.cpp/tools/slha_real_eval.cpp" \
    -L"$LLAMA_DIR/build/bin" -lllama -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"$LLAMA_DIR/build/bin" \
    -o "$EVAL_BIN"

run_arm() {
    local mode="$1"
    local json="$OUTPUT_DIR/$mode.eval.json"
    local logits="$OUTPUT_DIR/$mode.logits.f32"
    local log="$OUTPUT_DIR/$mode.log"
    local time_file="$OUTPUT_DIR/$mode.time"

    unset SLHA_EXTERNAL_K SLHA_KV_MODE SLHA_SCORE_MODE SLHA_SCORE_LAYERS
    unset SLHA_SCORE_ORACLE SLHA_ORACLE_METRICS_JSON SLHA_SCALE_FIT_JSON
    unset SLHA_RANK_DATASET_DIR SLHA_WEIGHTS_DIR SLHA_CODEC
    unset SLHA_CCOS SLHA_CCOS_BUDGET_BYTES SLHA_CCOS_IMPORTANCE_TEMPERATURE

    if [[ "$mode" == "external" ]]; then
        export SLHA_EXTERNAL_K=1
        export SLHA_KV_MODE=tilestore
        export SLHA_SCORE_MODE=replace
        export SLHA_SCORE_LAYERS=all
        export SLHA_WEIGHTS_DIR="$WEIGHTS_DIR"
        export SLHA_CODEC="$CODEC"
        if [[ "$CCOS" -eq 1 ]]; then
            export SLHA_CCOS=1
            [[ -z "$CCOS_BUDGET_BYTES" ]] || export SLHA_CCOS_BUDGET_BYTES="$CCOS_BUDGET_BYTES"
            [[ -z "$CCOS_IMPORTANCE_TEMPERATURE" ]] || \
                export SLHA_CCOS_IMPORTANCE_TEMPERATURE="$CCOS_IMPORTANCE_TEMPERATURE"
        fi
    fi

    echo "== running $mode =="
    set +e
    LC_ALL=C /usr/bin/time \
        -f 'max_rss_kb=%M\nelapsed_s=%e\nuser_s=%U\nsys_s=%S' \
        -o "$time_file" \
        "$EVAL_BIN" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --output-json "$json" \
        --logits-bin "$logits" \
        --max-tokens "$MAX_TOKENS" \
        --context-size "$CTX_SIZE" \
        --threads "$THREADS" \
        --gpu-layers "$GPU_LAYERS" \
        --cache-type-k "$CACHE_TYPE_K" \
        --cache-type-v "$CACHE_TYPE_V" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "ERROR: $mode evaluation failed with exit code $rc" >&2
        return "$rc"
    fi
    python3 -m json.tool "$json" >/dev/null
}

run_arm baseline
run_arm external

python3 "$REPO_ROOT/integration/llama.cpp/scripts/compare_real_eval.py" \
    --baseline-json "$OUTPUT_DIR/baseline.eval.json" \
    --external-json "$OUTPUT_DIR/external.eval.json" \
    --baseline-logits "$OUTPUT_DIR/baseline.logits.f32" \
    --external-logits "$OUTPUT_DIR/external.logits.f32" \
    --baseline-time "$OUTPUT_DIR/baseline.time" \
    --external-time "$OUTPUT_DIR/external.time" \
    --baseline-log "$OUTPUT_DIR/baseline.log" \
    --external-log "$OUTPUT_DIR/external.log" \
    --model "$MODEL" \
    --model-sha256 "$MODEL_SHA256" \
    --prompt-sha256 "$PROMPT_SHA256" \
    --slhav2-commit "$SLHA_COMMIT" \
    --llama-commit "$LLAMA_EXPECTED" \
    --output "$OUTPUT_DIR/comparison.json"

python3 - "$OUTPUT_DIR/comparison.json" "$CCOS" <<'PY'
import json, sys
p = sys.argv[1]
ccos_requested = sys.argv[2] == "1"
r = json.load(open(p))
valid = r.get("validity", {}).get("external_replace_valid")
if valid is not True:
    raise SystemExit(f"external SLHA replace summary is not valid: {valid!r}")
validity = r.get("validity", {})
backend = validity.get("external_backend")
if ccos_requested:
    if backend != "ccos_elastic":
        raise SystemExit(f"CCOS was requested but measured backend is {backend!r}")
    if validity.get("ccos_dense_no_cold") is not True:
        raise SystemExit("dense CCOS run observed a COLD slot")
    if validity.get("ccos_budget_failures") != 0:
        raise SystemExit(
            f"CCOS budget failures are non-zero: {validity.get('ccos_budget_failures')!r}"
        )
q = r["quality"]
pf = r["performance"]
mem = r["memory"]
print("== paired result ==")
print(f"token agreement     : {q['aligned_token_agreement_ratio']}")
print(f"common prefix       : {q['common_prefix_tokens']} tokens")
print(f"first divergence    : {q['first_divergence_index']}")
print(f"text exact match    : {q['text_exact_match']}")
print(f"logit relative L2   : {q['next_token_logits']['relative_l2'] if q['next_token_logits'] else None}")
print(f"baseline decode t/s : {pf['baseline_decode_tokens_per_second']}")
print(f"external decode t/s : {pf['external_decode_tokens_per_second']}")
print(f"baseline peak RSS KB: {mem['baseline_max_process_rss_kb']}")
print(f"external peak RSS KB: {mem['external_max_process_rss_kb']}")
print(f"external backend    : {validity.get('external_backend')}")
if ccos_requested:
    store = mem.get("external_slha_store") or {}
    print(f"CCOS peak resident  : {store.get('peak_resident_bytes')} bytes")
    print(f"CCOS peak offloaded : {store.get('peak_offloaded_bytes')} bytes")
    print(f"CCOS HOT/WARM/COLD  : {store.get('peak_hot_slots')}/{store.get('peak_warm_slots')}/{store.get('peak_cold_slots')}")
    print(f"CCOS compression ms : {pf.get('slha_compression_ms')}")
    print(f"CCOS score ms       : {pf.get('slha_score_ms')}")
    print(f"CCOS budget ms      : {pf.get('slha_budget_enforcement_ms')}")
print(f"report              : {p}")
PY
