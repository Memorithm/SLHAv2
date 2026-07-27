#!/usr/bin/env bash
# Train one SLHA projection (.slhw) per layer from collected K activations.
#
# Usage: WORK=/path/to/scratch ./train_layer_weights.sh [codec]
#
# Reads layer dumps from $WORK/calibration (produced by collect mode) and writes
# per-layer weights to $WORK/weights, plus a manifest.json describing the set.
set -euo pipefail

WORK="${WORK:-/tmp/slha-llama}"
CALIB_DIR="${CALIB_DIR:-$WORK/calibration}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$WORK/weights}"
CODEC="${1:-mixed}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../../..")"

if [ ! -d "$CALIB_DIR" ]; then
    echo "ERROR: calibration directory not found: $CALIB_DIR"
    echo "  Run collect mode first (e.g. build_and_roundtrip.sh collect with DATA_FILE=...)."
    exit 1
fi

mkdir -p "$WEIGHTS_DIR"

LLAMA_COMMIT="unknown"
if [ -f "$WORK/llama.cpp/.git/HEAD" ]; then
    LLAMA_COMMIT="$(cd "$WORK/llama.cpp" && git rev-parse HEAD)"
fi

MODEL_ID="${MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}/${MODEL_FILE:-qwen2.5-0.5b-instruct-q8_0.gguf}"

echo "== training per-layer SLHA projections =="
echo "  calibration : $CALIB_DIR"
echo "  weights out : $WEIGHTS_DIR"
echo "  codec       : $CODEC"
echo "  llama.cpp   : $LLAMA_COMMIT"

LAYERS_JSON='['
FIRST=1

for dump in "$CALIB_DIR"/layer-*-k.bin; do
    [ -e "$dump" ] || { echo "ERROR: no layer dumps in $CALIB_DIR"; exit 1; }

    basename_dumps="$(basename "$dump")"
    # layer-<id>-k.bin
    layer_id="${basename_dumps#layer-}"
    layer_id="${layer_id%-k.bin}"
    weight_name="$(printf "layer-%03d.slhw" "$layer_id")"
    weight_path="$WEIGHTS_DIR/$weight_name"

    if [ -f "$weight_path" ]; then
        echo "  layer $layer_id: $weight_path already exists, skipping"
        # Still include in manifest if not first.
        # Read dimension from existing dump for manifest.
        cols="$(python3 - "$dump" <<'PY'
import sys, struct
with open(sys.argv[1], "rb") as f:
    hdr = f.read(12)
    print(struct.unpack("<III", hdr)[2])
PY
)"
        if [ "$FIRST" -eq 1 ]; then FIRST=0; else LAYERS_JSON="$LAYERS_JSON,"; fi
        LAYERS_JSON="$LAYERS_JSON{\"layer\":$layer_id,\"file\":\"$weight_name\",\"input_dim\":$cols}"
        continue
    fi

    echo "  layer $layer_id: training from $dump -> $weight_path"

    tmpdir="$(mktemp -d)"
    cp "$dump" "$tmpdir/k.bin"

    cols="$(python3 - "$dump" <<'PY'
import sys, struct
with open(sys.argv[1], "rb") as f:
    hdr = f.read(12)
    print(struct.unpack("<III", hdr)[2])
PY
)"

    ( cd "$REPO_ROOT" && cargo run --release --example train_on_real_activations -- \
        --dump "$tmpdir" --out "$weight_path" >/dev/null 2>&1 ) || {
        echo "ERROR: training failed for layer $layer_id"
        rm -rf "$tmpdir"
        exit 1
    }

    rm -rf "$tmpdir"

    if [ ! -f "$weight_path" ]; then
        echo "ERROR: expected weight file missing: $weight_path"
        exit 1
    fi

    if [ "$FIRST" -eq 1 ]; then FIRST=0; else LAYERS_JSON="$LAYERS_JSON,"; fi
    LAYERS_JSON="$LAYERS_JSON{\"layer\":$layer_id,\"file\":\"$weight_name\",\"input_dim\":$cols}"
done

LAYERS_JSON="$LAYERS_JSON]"

NUM_LAYERS="$(echo "$LAYERS_JSON" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"

cat > "$WEIGHTS_DIR/manifest.json" <<EOF
{
  "format_version": 1,
  "model_identifier": "$MODEL_ID",
  "llama_cpp_commit": "$LLAMA_COMMIT",
  "num_layers": $NUM_LAYERS,
  "codec": "$CODEC",
  "layers": $LAYERS_JSON
}
EOF

echo
echo "== trained $NUM_LAYERS layer projections in $WEIGHTS_DIR =="
echo "  manifest: $WEIGHTS_DIR/manifest.json"
