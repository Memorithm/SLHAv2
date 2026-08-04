#!/usr/bin/env bash
# Build llama.cpp with SLHA K-cache integration and run perplexity.
#
# Usage: WORK=/path/to/scratch ./build_and_roundtrip.sh [mode]
#
<<<<<<< HEAD
# Modes: baseline, passthrough, roundtrip, collect, scorediag (default: passthrough)
=======
# Modes: baseline, passthrough, roundtrip, collect, shadow (default: passthrough)
>>>>>>> origin/master
set -euo pipefail

WORK="${WORK:-/tmp/slha-llama}"
# Pinned llama.cpp commit for the SLHA integration patches.
LLAMA_COMMIT="${LLAMA_COMMIT:-fdb1db877c526ec90f668eca1b858da5dba85560}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-1.5b-instruct-q8_0.gguf}"
CHUNKS="${CHUNKS:-12}"
THREADS="${THREADS:-4}"
MODE="${1:-passthrough}"
DATA_FILE="${DATA_FILE:-$WORK/wiki.test.raw}"
CALIB_DIR="${CALIB_DIR:-$WORK/calibration}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../../..")"

mkdir -p "$WORK"
cd "$WORK"

echo "== SLHA K-cache integration build =="
echo "  mode: $MODE"
echo "  work: $WORK"
echo "  llama.cpp commit: $LLAMA_COMMIT"

# 1. Build slha-c (libslha) if needed.
if [ "$MODE" != "baseline" ]; then
    echo "== building slha-c (libslha) =="
    ( cd "$REPO_ROOT" && cargo build --release -p slha-c ) || \
        echo "  (build slha-c from the SLHAv2 repo root: cargo build --release -p slha-c)"
fi

# 2. Clone + verify llama.cpp at the pinned commit.
if [ ! -d llama.cpp ]; then
    echo "== cloning llama.cpp @ $LLAMA_COMMIT =="
    git init llama.cpp
    (
        cd llama.cpp
        git remote add origin https://github.com/ggml-org/llama.cpp
        git fetch --depth=1 origin "$LLAMA_COMMIT"
        git checkout "$LLAMA_COMMIT"
    )
fi

# Verify the commit strictly.
ACTUAL_COMMIT=$(cd llama.cpp && git rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$LLAMA_COMMIT" ]; then
    echo "ERROR: llama.cpp commit mismatch"
    echo "  expected: $LLAMA_COMMIT"
    echo "  actual:   $ACTUAL_COMMIT"
    echo "  remove $WORK/llama.cpp and re-run to clone the pinned commit"
    exit 1
fi
echo "  llama.cpp commit verified: $ACTUAL_COMMIT"

# 3. Apply patches if not baseline.
if [ "$MODE" != "baseline" ]; then
<<<<<<< HEAD
    echo "== applying SLHA patches =="

    apply_patch_strict() {
        local patch_file="$1"
        local patch_name
        patch_name="$(basename "$patch_file")"

        echo "  checking $patch_name ..."
        if git apply --check "$patch_file" 2>/dev/null; then
            echo "  applying $patch_name ..."
            if ! patch --fuzz=0 -p1 < "$patch_file"; then
                echo "ERROR: $patch_name failed to apply"
                exit 1
            fi
        elif git apply --check -R "$patch_file" 2>/dev/null; then
            echo "  $patch_name already applied, skipping"
        else
            echo "ERROR: $patch_name does not apply cleanly to pinned llama.cpp commit"
            echo "       expected commit: $LLAMA_COMMIT"
            exit 1
        fi

        if find . -name "*.rej" -print -quit | grep -q .; then
            echo "ERROR: reject files found after applying $patch_name"
            find . -name "*.rej" -print
            exit 1
        fi
    }

    # Apply each patch in order from patches/ directory.
    for PATCH_FILE in "$REPO_ROOT/integration/llama.cpp/patches/"*.patch; do
        PATCH_BASENAME="$(basename "$PATCH_FILE")"
        if [ ! -f "$PATCH_FILE" ]; then
            echo "  no patches found"
            break
        fi
=======
    echo "== applying SLHA patch =="
    PATCH_FILE="$REPO_ROOT/integration/llama.cpp/patches/0001-slha-k-passthrough.patch"
    if [ ! -f "$PATCH_FILE" ]; then
        echo "ERROR: patch file not found: $PATCH_FILE"
        exit 1
    fi

    # Check if patch is already applied.
    if ! grep -q "SLHA_INTEGRATION" llama.cpp/src/llama-kv-cache.cpp 2>/dev/null; then
>>>>>>> origin/master
        cd llama.cpp
        apply_patch_strict "$PATCH_FILE"
        cd ..
<<<<<<< HEAD
    done

    echo "  patch status: fuzz=0 offset=0 rejects=0"
=======
    else
        echo "  patch already applied"
    fi
>>>>>>> origin/master

    # Copy the shim files.
    echo "== copying SLHA shim =="
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_llama.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_llama.cpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_replace_counters.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_replace_counters.cpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_layer_mask.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_layer_mask.cpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_score_scale.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_score_scale.cpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_scale_fit.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_scale_fit.cpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_score_oracle.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_score_oracle.cpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_oracle_metrics.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_oracle_metrics.cpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_rank_dataset.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_rank_dataset.cpp" llama.cpp/src/
fi

# 4. Build llama.cpp.
echo "== building llama.cpp ==="
CMAKE_FLAGS="-DGGML_NATIVE=ON -DLLAMA_CURL=OFF"
if [ "$MODE" != "baseline" ]; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DSLHA_INTEGRATION=ON"
    CMAKE_FLAGS="$CMAKE_FLAGS -DSLHA_INCLUDE_DIR=$REPO_ROOT/slha-c/include"
    CMAKE_FLAGS="$CMAKE_FLAGS -DSLHA_LIB=$REPO_ROOT/target/release/libslha.a"
fi

cmake -S llama.cpp -B llama.cpp/build $CMAKE_FLAGS >/dev/null
cmake --build llama.cpp/build -j --target llama-perplexity llama-cli

# 5. Fetch the model if needed.
if [ ! -f "$MODEL_FILE" ]; then
    echo "== downloading $MODEL_FILE =="
    python3 - "$MODEL_REPO" "$MODEL_FILE" "$WORK" <<'PY' || \
        echo "  download it manually into $WORK and re-run"
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3])
PY
fi

# 6. Dataset slice (default: WikiText-2 test; override with DATA_FILE).
if [ ! -f "$DATA_FILE" ]; then
  echo "== building default wikitext-2 test slice =="
  python3 - <<'PY' || echo "  provide DATA_FILE manually"
from datasets import load_dataset
d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
open("wiki.test.raw","w").write("\n".join(x for x in d["text"] if x.strip())[:120000])
PY
fi

# 7. Run perplexity.
echo "== running perplexity (mode=$MODE, $CHUNKS chunks, $THREADS threads) =="

# Set environment variables for SLHA modes.
if [ "$MODE" = "passthrough" ]; then
    export SLHA_KV_MODE=passthrough
elif [ "$MODE" = "roundtrip" ]; then
    export SLHA_KV_MODE=roundtrip
    export SLHA_WEIGHTS_DIR="$WORK/weights"
elif [ "$MODE" = "collect" ]; then
    export SLHA_KV_MODE=collect
    export SLHA_WEIGHTS_DIR="$CALIB_DIR"
<<<<<<< HEAD
elif [ "$MODE" = "scorediag" ]; then
    export SLHA_KV_MODE=scorediag
    export SLHA_WEIGHTS_DIR="$WORK/weights"
elif [ "$MODE" = "fused" ]; then
    export SLHA_KV_MODE=fused
=======
elif [ "$MODE" = "shadow" ]; then
    export SLHA_KV_MODE=tilestore
    export SLHA_SCORE_MODE=shadow
    export SLHA_WEIGHTS_DIR="$WORK/weights"
elif [ "$MODE" = "replace" ]; then
    export SLHA_KV_MODE=tilestore
    export SLHA_SCORE_MODE=replace
>>>>>>> origin/master
    export SLHA_WEIGHTS_DIR="$WORK/weights"
else
    unset SLHA_KV_MODE
fi

if [ "$MODE" = "roundtrip" ] || [ "$MODE" = "scorediag" ] || [ "$MODE" = "fused" ]; then
    OUTPUT_FILE="$WORK/${MODE}_${SLHA_CODEC:-mixed}_ppl.txt"
else
    OUTPUT_FILE="$WORK/${MODE}_ppl.txt"
fi
<<<<<<< HEAD
# Fused and scorediag operate on the standard (non-flash) attention path —
# the SLHA custom node is a GGML op on that path. Flash attention must be
# disabled explicitly (`-fa off`) or the fused/diag callback never runs.
FA_ARGS=""
if [ "$MODE" = "fused" ] || [ "$MODE" = "scorediag" ]; then
    FA_ARGS="-fa off"
fi

llama.cpp/build/bin/llama-perplexity \
    -m "$MODEL_FILE" -f "$DATA_FILE" --chunks "$CHUNKS" -t "$THREADS" $FA_ARGS \
    2>&1 | tee "$OUTPUT_FILE" | grep -E "Final estimate|PPL"
=======
FLASH_ATTN_FLAG=""
PARALLEL_FLAG=""
BATCH_FLAG=""
if [ "$MODE" = "shadow" ] || [ "$MODE" = "replace" ]; then
    # Shadow scoring and the direct compressed-score path require the non-flash
    # attention path so kq logits are materialised, and a single stream so the
    # tile store positions are contiguous.
    # --parallel 1 and --batch-size 512 ensure n_seq=1 so tile store positions
    # are contiguous within a single sequence (no multi-sequence collision).
    FLASH_ATTN_FLAG="--flash-attn off"
    PARALLEL_FLAG="--parallel 1"
    BATCH_FLAG="--batch-size 512"
fi

set +o pipefail  # Temporarily disable for grep filter
llama.cpp/build/bin/llama-perplexity \
    -m "$MODEL_FILE" -f "$DATA_FILE" --chunks "$CHUNKS" -t "$THREADS" $FLASH_ATTN_FLAG $PARALLEL_FLAG $BATCH_FLAG \
    2>&1 | tee "$OUTPUT_FILE" | grep -E "Final estimate|PPL" || true
set -o pipefail
>>>>>>> origin/master

echo
echo "Results written to $OUTPUT_FILE"
echo "Mode: $MODE"
<<<<<<< HEAD
echo "llama.cpp commit: $ACTUAL_COMMIT"
=======
echo "llama.cpp commit: $LLAMA_COMMIT"

# Collect-mode validation: emit a provenance manifest and enforce the
# non-finite policy on the freshly collected calibration dumps. Default policy
# is 'reject', so a collection that captured any non-finite (NaN / +/-Inf) row
# fails here with a non-zero status instead of silently poisoning training.
if [ "$MODE" = "collect" ]; then
    echo "== validating collected calibration (policy=${SLHA_CALIBRATION_NONFINITE_POLICY:-reject}) =="
    SHIM_DIR="$REPO_ROOT/integration/llama.cpp/shim"
    CAL_TMP="$(mktemp -d)"
    CALIBRATE_BIN="$CAL_TMP/slha_calibrate"
    if "${CXX:-g++}" -O2 -std=c++17 -I"$SHIM_DIR" \
            "$SHIM_DIR/slha_calibrate_cli.cpp" "$SHIM_DIR/slha_calibration.cpp" \
            -o "$CALIBRATE_BIN"; then
        REF_DIM="$(python3 - "$CALIB_DIR" <<'PY'
import sys, glob, struct, re, os
files = sorted(glob.glob(os.path.join(sys.argv[1], "layer-*-k.bin")),
               key=lambda p: int(re.search(r"layer-(\d+)-k", p).group(1)))
print(struct.unpack("<III", open(files[0], "rb").read(12))[2] if files else 0)
PY
)"
        N_LAYERS="$(python3 - "$CALIB_DIR" <<'PY'
import sys, glob, os
print(len(glob.glob(os.path.join(sys.argv[1], "layer-*-k.bin"))))
PY
)"
        MODEL_SHA="$(sha256sum "$MODEL_FILE" 2>/dev/null | cut -d' ' -f1)"
        DATA_SHA="$(sha256sum "$DATA_FILE" 2>/dev/null | cut -d' ' -f1)"
        TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        IMPL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
        set +e
        "$CALIBRATE_BIN" "$CALIB_DIR" \
            --policy "${SLHA_CALIBRATION_NONFINITE_POLICY:-reject}" \
            --expect-dim "$REF_DIM" \
            --expect-layers "$N_LAYERS" \
            --min-rows "${SLHA_CALIBRATION_MIN_ROWS:-1}" \
            --manifest "$CALIB_DIR/calibration_manifest.json" \
            --impl-commit "$IMPL_COMMIT" \
            --llama-commit "$LLAMA_COMMIT" \
            --model-id "$MODEL_REPO/$MODEL_FILE" \
            --model-sha "$MODEL_SHA" \
            --dataset-sha "$DATA_SHA" \
            --timestamp "$TS_UTC" \
            --command "build_and_roundtrip.sh collect"
        CAL_RC=$?
        set -e
        rm -rf "$CAL_TMP"
        echo "  manifest: $CALIB_DIR/calibration_manifest.json"
        if [ "$CAL_RC" -ne 0 ]; then
            echo "ERROR: calibration validation FAILED (rc=$CAL_RC). The collected dumps"
            echo "  contain non-finite rows or a structural defect under the current policy."
            echo "  Fix the source, re-collect, or (research only) re-run with"
            echo "  SLHA_CALIBRATION_NONFINITE_POLICY=drop-row."
            exit 1
        fi
        echo "  calibration validation PASSED"
    else
        echo "ERROR: failed to build slha_calibrate validator"
        rm -rf "$CAL_TMP"
        exit 1
    fi
fi

# Replace-mode validation: parse SLHA_REPLACE_SUMMARY and reject unless valid.
if [ "$MODE" = "replace" ]; then
    echo "== validating strict replace coverage =="
    SUMMARY_BLOCK=$(grep -A 30 "^SLHA_REPLACE_SUMMARY$" "$OUTPUT_FILE" || true)
    VALID=$(echo "$SUMMARY_BLOCK" | grep "^valid=" | head -1 | cut -d= -f2)
    ACTIVE_COVERAGE=$(echo "$SUMMARY_BLOCK" | grep "^active_coverage=" | head -1 | cut -d= -f2)
    CALLBACKS=$(echo "$SUMMARY_BLOCK" | grep "^callbacks=" | head -1 | cut -d= -f2)
    AE_V=$(echo "$SUMMARY_BLOCK" | grep "^active_expected_vectors=" | head -1 | cut -d= -f2)
    AR_V=$(echo "$SUMMARY_BLOCK" | grep "^active_replaced_vectors=" | head -1 | cut -d= -f2)
    AE_L=$(echo "$SUMMARY_BLOCK" | grep "^active_expected_logits=" | head -1 | cut -d= -f2)
    AR_L=$(echo "$SUMMARY_BLOCK" | grep "^active_replaced_logits=" | head -1 | cut -d= -f2)
    PAD_V=$(echo "$SUMMARY_BLOCK" | grep "^padding_vectors=" | head -1 | cut -d= -f2)
    PAD_L=$(echo "$SUMMARY_BLOCK" | grep "^padding_logits=" | head -1 | cut -d= -f2)
    INACT_V=$(echo "$SUMMARY_BLOCK" | grep "^inactive_stream_vectors=" | head -1 | cut -d= -f2)
    INACT_L=$(echo "$SUMMARY_BLOCK" | grep "^inactive_stream_logits=" | head -1 | cut -d= -f2)
    FAILED_V=$(echo "$SUMMARY_BLOCK" | grep "^failed_vectors=" | head -1 | cut -d= -f2)
    FALLBACK_V=$(echo "$SUMMARY_BLOCK" | grep "^fallback_vectors=" | head -1 | cut -d= -f2)
    N_STREAM=$(echo "$SUMMARY_BLOCK" | grep "^n_stream=" | head -1 | cut -d= -f2)

    if [ -z "$VALID" ]; then
        echo "ERROR: SLHA_REPLACE_SUMMARY not found in output. Replace path may not have executed."
        exit 1
    fi

    echo "  callbacks=$CALLBACKS"
    echo "  active_expected_vectors=$AE_V"
    echo "  active_replaced_vectors=$AR_V"
    echo "  active_expected_logits=$AE_L"
    echo "  active_replaced_logits=$AR_L"
    echo "  padding_vectors=$PAD_V"
    echo "  padding_logits=$PAD_L"
    echo "  inactive_stream_vectors=$INACT_V"
    echo "  inactive_stream_logits=$INACT_L"
    echo "  failed_vectors=$FAILED_V"
    echo "  fallback_vectors=$FALLBACK_V"
    echo "  n_stream=$N_STREAM"
    echo "  active_coverage=$ACTIVE_COVERAGE"
    echo "  valid=$VALID"

    if [ "$VALID" != "true" ]; then
        echo "ERROR: strict replace validation FAILED (valid=$VALID). A valid replace run requires"
        echo "  active_replaced_vectors == active_expected_vectors,"
        echo "  active_replaced_logits == active_expected_logits,"
        echo "  failed_vectors == 0, fallback_vectors == 0,"
        echo "  n_stream == 1, active_coverage == 1, valid == true"
        exit 1
    fi

    if [ "$ACTIVE_COVERAGE" != "1" ] && [ "$ACTIVE_COVERAGE" != "1.0" ]; then
        echo "ERROR: strict replace coverage FAILED (active_coverage=$ACTIVE_COVERAGE). Expected 1.0"
        exit 1
    fi

    if [ "$N_STREAM" != "1" ]; then
        echo "ERROR: strict replace n_stream FAILED (n_stream=$N_STREAM). Expected 1"
        exit 1
    fi

    echo "  strict replace validation PASSED"
fi
>>>>>>> origin/master
