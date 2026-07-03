#!/usr/bin/env bash
# Phase 2 — reproducible llama.cpp baseline for the SLHA integration.
#
# Clones llama.cpp at a pinned tag, builds it CPU-only, fetches a small model
# and a WikiText-2 slice, and runs llama-perplexity to establish the baseline
# perplexity the SLHA K-cache path is measured against (see README.md).
#
# This script does NOT patch llama.cpp — the K-cache interception is documented
# as staged work in README.md. It produces the baseline anchor and builds the
# slha-c bridge (libslha) that the patch will link against.
#
# Usage:  WORK=/path/to/scratch ./build_and_baseline.sh
set -euo pipefail

WORK="${WORK:-/tmp/slha-llama}"
LLAMA_TAG="${LLAMA_TAG:-b9860}"          # pinned; bump deliberately
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-0.5b-instruct-q8_0.gguf}"
CHUNKS="${CHUNKS:-12}"                   # small: tuned for a 4-core CPU
THREADS="${THREADS:-4}"

mkdir -p "$WORK"
cd "$WORK"

# 1. Build the SLHA C bridge (libslha) — the codec path the patch links to.
echo "== building slha-c (libslha) =="
( cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo .)" \
    && cargo build --release -p slha-c ) || \
  echo "  (build slha-c from the SLHAv2 repo root: cargo build --release -p slha-c)"

# 2. Clone + build llama.cpp CPU-only at the pinned tag.
if [ ! -d llama.cpp ]; then
  echo "== cloning llama.cpp @ $LLAMA_TAG =="
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp
fi
echo "== building llama.cpp (CPU) =="
cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null
cmake --build llama.cpp/build -j --target llama-perplexity llama-cli

# 3. Fetch the model (needs huggingface_hub; falls back to a manual hint).
if [ ! -f "$MODEL_FILE" ]; then
  echo "== downloading $MODEL_FILE =="
  python3 - "$MODEL_REPO" "$MODEL_FILE" "$WORK" <<'PY' || \
    echo "  download it manually into $WORK and re-run"
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3])
PY
fi

# 4. WikiText-2 test slice.
if [ ! -f wiki.test.raw ]; then
  echo "== building wikitext-2 slice =="
  python3 - <<'PY' || echo "  provide wiki.test.raw manually"
from datasets import load_dataset
d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
open("wiki.test.raw","w").write("\n".join(x for x in d["text"] if x.strip())[:120000])
PY
fi

# 5. Baseline perplexity.
echo "== baseline perplexity ($CHUNKS chunks, $THREADS threads) =="
llama.cpp/build/bin/llama-perplexity \
  -m "$MODEL_FILE" -f wiki.test.raw --chunks "$CHUNKS" -t "$THREADS" \
  2>&1 | tee baseline_ppl.txt | grep -E "Final estimate|PPL"

echo
echo "Baseline written to $WORK/baseline_ppl.txt"
echo "Next: apply the K-cache patch (README.md, 'The patch seam') and re-run"
echo "this perplexity command to measure Δppl for the mixed / mix3 codecs."
