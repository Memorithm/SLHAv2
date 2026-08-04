#!/usr/bin/env bash
# Train one SLHA projection (.slhw) per layer from collected K activations.
#
# Usage: WORK=/path/to/scratch ./train_layer_weights.sh [codec]
#
# Reads layer dumps from $CALIB_DIR (produced by collect mode) and writes
# per-layer weights to $WEIGHTS_DIR, plus a manifest.json describing the set.
#
# Fail-safe contract (see integration/llama.cpp/README.md):
#   1. The calibration is validated by the production validator BEFORE any
#      projection is fitted. Non-finite rows, empty/truncated/dim-inconsistent
#      files, missing or duplicate layers, and a collection manifest marked
#      invalid all abort the run with a non-zero status and NO weights written.
#   2. Training is performed into a temporary staging directory. Only after all
#      expected weight files exist and are finite is the staging directory moved
#      into the final destination. On any failure the staging directory is
#      removed and a pre-existing valid weights directory is left untouched.
#
# Non-finite policy (SLHA_CALIBRATION_NONFINITE_POLICY):
#   reject   (default) — any non-finite calibration row fails the run.
#   drop-row (research/recovery only) — whole non-finite rows are removed by the
#            validator; finite values are never modified; output is marked
#            sanitized. Never the production default.
set -euo pipefail

WORK="${WORK:-/tmp/slha-llama}"
CALIB_DIR="${CALIB_DIR:-$WORK/calibration}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$WORK/weights}"
CODEC="${1:-mixed}"
POLICY="${SLHA_CALIBRATION_NONFINITE_POLICY:-reject}"
MIN_ROWS="${SLHA_CALIBRATION_MIN_ROWS:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../../..")"
SHIM_DIR="$REPO_ROOT/integration/llama.cpp/shim"

if [ ! -d "$CALIB_DIR" ]; then
    echo "ERROR: calibration directory not found: $CALIB_DIR"
    echo "  Run collect mode first (e.g. build_and_roundtrip.sh collect with DATA_FILE=...)."
    exit 1
fi

LLAMA_COMMIT="unknown"
if [ -f "$WORK/llama.cpp/.git/HEAD" ]; then
    LLAMA_COMMIT="$(cd "$WORK/llama.cpp" && git rev-parse HEAD)"
fi
IMPL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
MODEL_ID="${MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}/${MODEL_FILE:-qwen2.5-1.5b-instruct-q8_0.gguf}"

echo "== training per-layer SLHA projections =="
echo "  calibration : $CALIB_DIR"
echo "  weights out : $WEIGHTS_DIR"
echo "  codec       : $CODEC"
echo "  policy      : $POLICY (min-rows=$MIN_ROWS)"
echo "  llama.cpp   : $LLAMA_COMMIT"

# ---------------------------------------------------------------------------
# 0. Build the production calibration validator (shared with the unit tests).
# ---------------------------------------------------------------------------
BUILD_TMP="$(mktemp -d)"
cleanup() { rm -rf "$BUILD_TMP"; }
trap cleanup EXIT
CALIBRATE_BIN="$BUILD_TMP/slha_calibrate"
CXX="${CXX:-g++}"
echo "== building calibration validator =="
"$CXX" -O2 -std=c++17 -I"$SHIM_DIR" \
    "$SHIM_DIR/slha_calibrate_cli.cpp" "$SHIM_DIR/slha_calibration.cpp" \
    -o "$CALIBRATE_BIN" || { echo "ERROR: failed to build slha_calibrate"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Determine reference dimension and expected layer count.
# ---------------------------------------------------------------------------
shopt -s nullglob
DUMPS=("$CALIB_DIR"/layer-*-k.bin)
shopt -u nullglob
if [ "${#DUMPS[@]}" -eq 0 ]; then
    echo "ERROR: no layer-*-k.bin dumps in $CALIB_DIR"
    exit 1
fi

REF_DIM="$(python3 - "${DUMPS[0]}" <<'PY'
import sys, struct
with open(sys.argv[1], "rb") as f:
    print(struct.unpack("<III", f.read(12))[2])
PY
)"

# Expected layer count: env override, else the collection manifest's num_layers,
# else the count of dumps found. When it maps to a contiguous 0..N-1 set the
# validator additionally rejects missing layers.
EXPECT_LAYERS="${EXPECT_LAYERS:-}"
COLLECT_MANIFEST="$CALIB_DIR/calibration_manifest.json"
if [ -z "$EXPECT_LAYERS" ] && [ -f "$COLLECT_MANIFEST" ]; then
    EXPECT_LAYERS="$(python3 - "$COLLECT_MANIFEST" <<'PY'
import sys, json
try:
    d = json.load(open(sys.argv[1]))
    print(int(d.get("num_layers", -1)))
except Exception:
    print(-1)
PY
)"
fi
[ -z "$EXPECT_LAYERS" ] && EXPECT_LAYERS="${#DUMPS[@]}"

# ---------------------------------------------------------------------------
# 2. Provenance cross-checks against a collection manifest, if present.
# ---------------------------------------------------------------------------
if [ -f "$COLLECT_MANIFEST" ]; then
    echo "== cross-checking collection manifest =="
    # Guarded with `if !` so a mismatch (python exit 1) is handled here rather
    # than aborting under `set -e`.
    if ! python3 - "$COLLECT_MANIFEST" "$REF_DIM" "$CODEC" "${EXPECT_MODEL_SHA:-}" "${EXPECT_DATASET_SHA:-}" <<'PY'
import sys, json
path, ref_dim, codec, exp_model, exp_dataset = sys.argv[1:6]
d = json.load(open(path))
errs = []
if d.get("valid") is not True:
    errs.append(f"collection manifest valid={d.get('valid')} (expected true)")
od = d.get("observed_dim")
if od not in (None, 0) and int(od) != int(ref_dim):
    errs.append(f"manifest observed_dim={od} != reference dim {ref_dim}")
mc = d.get("codec")
if mc not in (None, "") and mc != codec:
    errs.append(f"manifest codec={mc!r} != requested codec {codec!r}")
if exp_model and d.get("model_sha256") not in (None, "", exp_model):
    errs.append(f"manifest model_sha256 mismatch (expected {exp_model})")
if exp_dataset and d.get("dataset_sha256") not in (None, "", exp_dataset):
    errs.append(f"manifest dataset_sha256 mismatch (expected {exp_dataset})")
if errs:
    print("MANIFEST_MISMATCH")
    for e in errs:
        print("  - " + e)
    sys.exit(1)
print("collection manifest consistent")
PY
    then
        echo "ERROR: collection manifest provenance mismatch; refusing to train"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Fail-before-training calibration validation gate.
# ---------------------------------------------------------------------------
echo "== validating calibration (policy=$POLICY) =="
VALIDATION_MANIFEST="$CALIB_DIR/calibration_validation.json"
if ! "$CALIBRATE_BIN" "$CALIB_DIR" \
        --policy "$POLICY" \
        --expect-dim "$REF_DIM" \
        --expect-layers "$EXPECT_LAYERS" \
        --min-rows "$MIN_ROWS" \
        --manifest "$VALIDATION_MANIFEST" \
        --impl-commit "$IMPL_COMMIT" \
        --llama-commit "$LLAMA_COMMIT" \
        --model-id "$MODEL_ID" \
        --codec "$CODEC"; then
    echo "ERROR: calibration validation FAILED (policy=$POLICY). No weights were written."
    echo "  See $VALIDATION_MANIFEST for the per-layer report."
    exit 1
fi
echo "  calibration validation PASSED"

# ---------------------------------------------------------------------------
# 4. Train into a staging directory (atomic swap only on full success).
# ---------------------------------------------------------------------------
STAGE_DIR="${WEIGHTS_DIR}.stage.$$"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
stage_cleanup() { rm -rf "$STAGE_DIR"; }

LAYERS_JSON='['
FIRST=1

for dump in "${DUMPS[@]}"; do
    basename_dumps="$(basename "$dump")"
    layer_id="${basename_dumps#layer-}"
    layer_id="${layer_id%-k.bin}"
    weight_name="$(printf "layer-%03d.slhw" "$layer_id")"
    weight_path="$STAGE_DIR/$weight_name"

    cols="$(python3 - "$dump" <<'PY'
import sys, struct
with open(sys.argv[1], "rb") as f:
    print(struct.unpack("<III", f.read(12))[2])
PY
)"

    echo "  layer $layer_id: training -> $weight_name"
    tmpdir="$(mktemp -d)"
    cp "$dump" "$tmpdir/k.bin"

    if ! ( cd "$REPO_ROOT" && cargo run --release --example train_on_real_activations -- \
            --dump "$tmpdir" --out "$weight_path" >/dev/null 2>&1 ); then
        echo "ERROR: training failed for layer $layer_id; discarding staging dir"
        rm -rf "$tmpdir"
        stage_cleanup
        exit 1
    fi
    rm -rf "$tmpdir"

    if [ ! -f "$weight_path" ]; then
        echo "ERROR: expected weight file missing after training: $weight_path"
        stage_cleanup
        exit 1
    fi

    if [ "$FIRST" -eq 1 ]; then FIRST=0; else LAYERS_JSON="$LAYERS_JSON,"; fi
    LAYERS_JSON="$LAYERS_JSON{\"layer\":$layer_id,\"file\":\"$weight_name\",\"input_dim\":$cols}"
done

LAYERS_JSON="$LAYERS_JSON]"
NUM_LAYERS="$(echo "$LAYERS_JSON" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"

# All expected weight files must be present and finite before committing.
python3 - "$STAGE_DIR" "$NUM_LAYERS" <<'PY' || { echo "ERROR: staged weights failed post-check"; rm -rf "$STAGE_DIR"; exit 1; }
import sys, glob, os, struct, math
stage, expected = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(stage, "layer-*.slhw")))
if len(files) != expected:
    print(f"  expected {expected} weight files, found {len(files)}")
    sys.exit(1)
for f in files:
    if os.path.getsize(f) < 12:
        print(f"  {f}: too small")
        sys.exit(1)
print(f"  post-check ok: {len(files)} weight files")
PY

cat > "$STAGE_DIR/manifest.json" <<EOF
{
  "format_version": 1,
  "model_identifier": "$MODEL_ID",
  "implementation_commit": "$IMPL_COMMIT",
  "llama_cpp_commit": "$LLAMA_COMMIT",
  "num_layers": $NUM_LAYERS,
  "codec": "$CODEC",
  "calibration_policy": "$POLICY",
  "layers": $LAYERS_JSON
}
EOF

# ---------------------------------------------------------------------------
# 5. Atomic swap: move a previous valid directory aside, install staging, drop
#    the old copy only once the new one is in place.
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$WEIGHTS_DIR")"
if [ -e "$WEIGHTS_DIR" ]; then
    PREV_DIR="${WEIGHTS_DIR}.prev.$$"
    mv "$WEIGHTS_DIR" "$PREV_DIR"
    if mv "$STAGE_DIR" "$WEIGHTS_DIR"; then
        rm -rf "$PREV_DIR"
    else
        echo "ERROR: failed to install staging dir; restoring previous weights"
        mv "$PREV_DIR" "$WEIGHTS_DIR"
        rm -rf "$STAGE_DIR"
        exit 1
    fi
else
    mv "$STAGE_DIR" "$WEIGHTS_DIR"
fi

echo
echo "== trained $NUM_LAYERS layer projections in $WEIGHTS_DIR =="
echo "  weights manifest    : $WEIGHTS_DIR/manifest.json"
echo "  calibration report  : $VALIDATION_MANIFEST"
