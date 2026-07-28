#!/usr/bin/env bash
# Build llama.cpp with SLHA K-cache passthrough integration and run perplexity.
#
# Usage: WORK=/path/to/scratch ./build_and_roundtrip.sh [mode]
#
# Modes: baseline, passthrough, roundtrip, collect, scorediag (default: passthrough)
set -euo pipefail

WORK="${WORK:-/tmp/slha-llama}"
LLAMA_TAG="${LLAMA_TAG:-b9860}"
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
echo "  llama.cpp tag: $LLAMA_TAG"

# 1. Build slha-c (libslha) if needed.
if [ "$MODE" != "baseline" ]; then
    echo "== building slha-c (libslha) =="
    ( cd "$REPO_ROOT" && cargo build --release -p slha-c ) || \
        echo "  (build slha-c from the SLHAv2 repo root: cargo build --release -p slha-c)"
fi

# 2. Clone + build llama.cpp.
if [ ! -d llama.cpp ]; then
    echo "== cloning llama.cpp @ $LLAMA_TAG =="
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp
fi

# Verify the commit.
LLAMA_COMMIT=$(cd llama.cpp && git rev-parse HEAD)
echo "  llama.cpp commit: $LLAMA_COMMIT"

# 3. Apply patches if not baseline.
if [ "$MODE" != "baseline" ]; then
    echo "== applying SLHA patches =="
    
    # Apply each patch in order from patches/ directory.
    for PATCH_FILE in "$REPO_ROOT/integration/llama.cpp/patches/"*.patch; do
        PATCH_BASENAME="$(basename "$PATCH_FILE")"
        if [ ! -f "$PATCH_FILE" ]; then
            echo "  no patches found"
            break
        fi
        echo "  applying $PATCH_BASENAME ..."
        cd llama.cpp
        patch -p1 -N < "$PATCH_FILE" 2>/dev/null || echo "  (already applied or skipped)"
        cd ..
    done
    
    # Copy the shim files.
    echo "== copying SLHA shim =="
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_llama.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_llama.cpp" llama.cpp/src/
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
elif [ "$MODE" = "scorediag" ]; then
    export SLHA_KV_MODE=scorediag
    export SLHA_WEIGHTS_DIR="$WORK/weights"
else
    unset SLHA_KV_MODE
fi

if [ "$MODE" = "roundtrip" ] || [ "$MODE" = "scorediag" ]; then
    OUTPUT_FILE="$WORK/${MODE}_${SLHA_CODEC:-mixed}_ppl.txt"
else
    OUTPUT_FILE="$WORK/${MODE}_ppl.txt"
fi
llama.cpp/build/bin/llama-perplexity \
    -m "$MODEL_FILE" -f "$DATA_FILE" --chunks "$CHUNKS" -t "$THREADS" \
    2>&1 | tee "$OUTPUT_FILE" | grep -E "Final estimate|PPL"

echo
echo "Results written to $OUTPUT_FILE"
echo "Mode: $MODE"
echo "llama.cpp commit: $LLAMA_COMMIT"
