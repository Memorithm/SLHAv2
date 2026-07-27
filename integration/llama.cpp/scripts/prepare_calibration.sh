#!/usr/bin/env bash
# Prepare a calibration text corpus separate from the WikiText-2 test slice
# used for perplexity evaluation. The corpus must not overlap the evaluation
# data to avoid optimistic round-trip measurements.
set -euo pipefail

WORK="${WORK:-/tmp/slha-llama}"
CALIB_FILE="${CALIB_FILE:-$WORK/wiki.train.raw}"
MAX_CHARS="${MAX_CHARS:-120000}"

mkdir -p "$WORK"

if [ -f "$CALIB_FILE" ]; then
    echo "Calibration file already exists: $CALIB_FILE"
    exit 0
fi

echo "== building calibration corpus (WikiText-2 train, ${MAX_CHARS} chars) =="
python3 - "$CALIB_FILE" "$MAX_CHARS" <<'PY'
import sys
from datasets import load_dataset
path = sys.argv[1]
max_chars = int(sys.argv[2])
d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
text = "\n".join(x for x in d["text"] if x.strip())
open(path, "w").write(text[:max_chars])
print(f"  wrote {path} ({len(text[:max_chars])} chars)")
PY
