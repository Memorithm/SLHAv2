#!/usr/bin/env bash
# Build llama.cpp with SLHA K-cache passthrough integration and run perplexity.
#
# Usage: WORK=/path/to/scratch ./build_and_roundtrip.sh [mode]
#
# Modes: baseline, passthrough, roundtrip, collect (default: passthrough)
set -euo pipefail

WORK="${WORK:-/tmp/slha-llama}"
LLAMA_TAG="${LLAMA_TAG:-b9860}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-0.5b-instruct-q8_0.gguf}"
CHUNKS="${CHUNKS:-12}"
THREADS="${THREADS:-4}"
MODE="${1:-passthrough}"

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

# 3. Apply the patch if not baseline.
if [ "$MODE" != "baseline" ]; then
    echo "== applying SLHA patch =="
    PATCH_FILE="$REPO_ROOT/integration/llama.cpp/patches/0001-slha-k-passthrough.patch"
    if [ ! -f "$PATCH_FILE" ]; then
        echo "ERROR: patch file not found: $PATCH_FILE"
        exit 1
    fi
    
    # Check if patch is already applied.
    if ! grep -q "SLHA_INTEGRATION" llama.cpp/src/llama-kv-cache.cpp 2>/dev/null; then
        cd llama.cpp
        patch -p1 < "$PATCH_FILE"
        cd ..
    else
        echo "  patch already applied"
    fi
    
    # Copy the shim files.
    echo "== copying SLHA shim =="
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_llama.hpp" llama.cpp/src/
    cp "$REPO_ROOT/integration/llama.cpp/shim/slha_llama.cpp" llama.cpp/src/
fi

# 4. Build llama.cpp.
echo "== building llama.cpp =="
CMAKE_FLAGS="-DGGML_NATIVE=ON -DLLAMA_CURL=OFF"
if [ "$MODE" != "baseline" ]; then
    CMAKE_FLAGS="$CMAKE_FLAGS -DSLHA_INTEGRATION=ON"
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

# 6. WikiText-2 test slice.
if [ ! -f wiki.test.raw ]; then
    echo "== building wikitext-2 slice =="
    python3 - <<'PY' || echo "  provide wiki.test.raw manually"
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
    export SLHA_WEIGHTS_DIR="$WORK/weights"
else
    unset SLHA_KV_MODE
fi

OUTPUT_FILE="$WORK/${MODE}_ppl.txt"
llama.cpp/build/bin/llama-perplexity \
    -m "$MODEL_FILE" -f wiki.test.raw --chunks "$CHUNKS" -t "$THREADS" \
    2>&1 | tee "$OUTPUT_FILE" | grep -E "Final estimate|PPL"

echo
echo "Results written to $OUTPUT_FILE"
echo "Mode: $MODE"
echo "llama.cpp commit: $LLAMA_COMMIT"
