#!/usr/bin/env bash
# Production-linked trainer atomicity tests (items 12 and 13).
#
# Drives the real train_layer_weights.sh against a deliberately invalid
# calibration set. Because the fail-before-training gate rejects the calibration
# up front, no projection is ever fitted (cargo is not invoked), so these run
# fast and deterministically. They assert:
#   12. a failed run creates no partial final weights directory;
#   13. a pre-existing valid weights directory is left byte-for-byte untouched.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER="$SCRIPT_DIR/../scripts/train_layer_weights.sh"

fail=0
pass() { echo "  test: $1 ... ok"; }
bad()  { echo "  test: $1 ... FAILED"; fail=1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CALIB="$WORK/calibration"
mkdir -p "$CALIB"

# Craft two dumps; layer 1 carries a NaN so the default reject policy fails.
python3 - "$CALIB" <<'PY'
import sys, os, struct, math
d = sys.argv[1]
def write(layer, rows, cols, vals):
    with open(os.path.join(d, f"layer-{layer}-k.bin"), "wb") as f:
        f.write(struct.pack("<III", 0x534C4841, rows, cols))
        f.write(struct.pack("<%df" % (rows * cols), *vals))
write(0, 2, 4, [1,2,3,4, 5,6,7,8])
write(1, 2, 4, [1,2,3,4, 5, float("nan"), 7, 8])
PY

echo "=== SLHA trainer atomicity tests ==="

# ---- Test 12: no partial final weights on gate failure ----
WEIGHTS="$WORK/weights_t12"
CALIB_DIR="$CALIB" WEIGHTS_DIR="$WEIGHTS" WORK="$WORK" \
    bash "$TRAINER" mixed >"$WORK/t12.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] && [ ! -e "$WEIGHTS" ]; then
    pass "failed run creates no partial final weights (exit=$rc, no weights dir)"
else
    bad "failed run creates no partial final weights (exit=$rc, weights_present=$([ -e "$WEIGHTS" ] && echo yes || echo no))"
    sed 's/^/    /' "$WORK/t12.log"
fi

# ---- Test 13: pre-existing valid weights untouched on gate failure ----
WEIGHTS2="$WORK/weights_t13"
mkdir -p "$WEIGHTS2"
printf 'PRIOR-VALID-WEIGHTS' > "$WEIGHTS2/layer-000.slhw"
printf '{"format_version":1,"note":"prior"}' > "$WEIGHTS2/manifest.json"
before="$(cd "$WEIGHTS2" && find . -type f -exec sha256sum {} \; | sort)"

CALIB_DIR="$CALIB" WEIGHTS_DIR="$WEIGHTS2" WORK="$WORK" \
    bash "$TRAINER" mixed >"$WORK/t13.log" 2>&1
rc=$?
after="$(cd "$WEIGHTS2" && find . -type f -exec sha256sum {} \; | sort)"
if [ "$rc" -ne 0 ] && [ "$before" = "$after" ]; then
    pass "pre-existing valid weights untouched after failed retraining (exit=$rc)"
else
    bad "pre-existing valid weights untouched (exit=$rc, changed=$([ "$before" = "$after" ] && echo no || echo yes))"
    sed 's/^/    /' "$WORK/t13.log"
fi

# ---- Sanity: no leftover staging directories anywhere under WORK ----
if find "$WORK" -maxdepth 1 -name 'weights*.stage.*' -o -name 'weights*.prev.*' 2>/dev/null | grep -q .; then
    bad "no leftover staging/prev directories"
else
    pass "no leftover staging/prev directories"
fi

if [ "$fail" -eq 0 ]; then
    echo "=== trainer atomicity tests: ALL PASS ==="
    exit 0
else
    echo "=== trainer atomicity tests: FAILURES PRESENT ==="
    exit 1
fi
