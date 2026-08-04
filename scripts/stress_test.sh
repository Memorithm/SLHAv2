#!/usr/bin/env bash
# SLHA v2 — massive stress / acceptance test harness.
#
# Runs the full quality gate (fmt, clippy, build, tests, doc, benches,
# cross-compile), exercises EVERY example, checks output determinism, can soak
# the hot kernel, and writes a timestamped Markdown + JSON report under
# target/stress/ so a run is auditable after the fact.
#
# Usage:
#   scripts/stress_test.sh             # full suite (debug + release)
#   scripts/stress_test.sh --quick     # skip the separate release workspace
#                                      # build + release tests + soak (examples
#                                      # still run in release)
#   scripts/stress_test.sh --soak N    # additionally run the throughput example N times
#   scripts/stress_test.sh --no-cross  # skip the aarch64 cross-compile step
#   scripts/stress_test.sh -h          # this help
#
# Exit code: 0 iff every non-skipped step passed.

set -uo pipefail

# ── locate repo root (script lives in scripts/) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── options ──────────────────────────────────────────────────────────────────
QUICK=0; SOAK=0; DO_CROSS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)    QUICK=1 ;;
    --soak)     SOAK="${2:-20}"; shift ;;
    --no-cross) DO_CROSS=0 ;;
    -h|--help)  grep '^#' "$0" | sed 's/^#\{1,\} \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try -h)"; exit 2 ;;
  esac
  shift
done

# ── colors (tty only) ────────────────────────────────────────────────────────
if [ -t 1 ]; then
  R=$'\033[0;31m'; G=$'\033[0;32m'; Y=$'\033[1;33m'; C=$'\033[0;36m'; B=$'\033[1m'; N=$'\033[0m'
else R=; G=; Y=; C=; B=; N=; fi

# ── report files ─────────────────────────────────────────────────────────────
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$REPO_ROOT/target/stress"
LOG_DIR="$OUT_DIR/logs-$STAMP"
REPORT_MD="$OUT_DIR/report-$STAMP.md"
SUMMARY_JSON="$OUT_DIR/summary-$STAMP.json"
mkdir -p "$LOG_DIR"

PASS=0; FAIL=0; SKIP=0
declare -a ROWS=()
declare -a FAILED_NAMES=()

have() { command -v "$1" >/dev/null 2>&1; }

TIMEOUT_BIN=""
if have timeout; then TIMEOUT_BIN="timeout"; elif have gtimeout; then TIMEOUT_BIN="gtimeout"; fi
run_to() { local secs="$1"; shift; if [ -n "$TIMEOUT_BIN" ]; then "$TIMEOUT_BIN" "${secs}s" "$@"; else "$@"; fi; }

banner() { echo; echo "${B}$*${N}"; printf '%s' "$B"; printf '─%.0s' $(seq 1 62); printf '%s\n' "$N"; }

# step "Name" [timeout_secs] -- cmd...
step() {
  local name="$1"; shift
  local to=600
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then to="$1"; shift; fi
  [ "${1:-}" = "--" ] && shift
  local slug; slug="$(printf '%s' "$name" | tr ' /:' '___' | tr -cd '[:alnum:]_.-')"
  local log="$LOG_DIR/$slug.log"
  printf "%s▶ %-50s%s" "$C" "$name" "$N"
  local t0 t1 rc secs
  t0=$(date +%s)
  run_to "$to" "$@" >"$log" 2>&1; rc=$?
  t1=$(date +%s); secs=$((t1 - t0))
  if [ $rc -eq 0 ]; then
    printf "%s PASS%s (%ss)\n" "$G" "$N" "$secs"; PASS=$((PASS+1)); ROWS+=("✅|$name|${secs}s|")
  elif [ $rc -eq 124 ]; then
    printf "%s TIMEOUT%s (>%ss)\n" "$R" "$N" "$to"; FAIL=$((FAIL+1)); ROWS+=("⏰|$name|>${to}s|timeout"); FAILED_NAMES+=("$name")
    tail -8 "$log" | sed 's/^/    /'
  else
    printf "%s FAIL%s (rc=%s, %ss)\n" "$R" "$N" "$rc" "$secs"; FAIL=$((FAIL+1)); ROWS+=("❌|$name|${secs}s|rc=$rc"); FAILED_NAMES+=("$name")
    tail -12 "$log" | sed 's/^/    /'
  fi
}

skip() { local name="$1"; local why="${2:-}"; printf "%s▷ %-50s SKIP%s %s\n" "$Y" "$name" "$N" "$why"; SKIP=$((SKIP+1)); ROWS+=("⏭️|$name||$why"); }

# ── preamble ─────────────────────────────────────────────────────────────────
banner "SLHA v2 — Stress / Acceptance Harness"
echo "  repo:   $REPO_ROOT"
echo "  host:   $(uname -srm)   cpus=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
echo "  rustc:  $(rustc --version 2>/dev/null || echo 'MISSING')"
echo "  cargo:  $(cargo --version 2>/dev/null || echo 'MISSING')"
echo "  mode:   $([ $QUICK -eq 1 ] && echo quick || echo full)   soak=$SOAK   cross=$DO_CROSS"
echo "  report: $REPORT_MD"
if ! have cargo; then echo "${R}cargo not found — install Rust: https://rustup.rs${N}"; exit 1; fi

# ── 1. static quality gate ──────────────────────────────────────────────────
banner "1. Static quality gate"
step "fmt --check"                    120 -- cargo fmt --all --check
step "clippy -D warnings (all-targets)" 600 -- cargo clippy --workspace --all-targets --all-features -- -D warnings

# ── 2. build ────────────────────────────────────────────────────────────────
banner "2. Build"
step "build debug (all targets)"      600 -- cargo build --workspace --all-targets --all-features
if [ $QUICK -eq 0 ]; then
  step "build release"                600 -- cargo build --workspace --release --all-features
fi

# ── 3. tests / docs / benches ───────────────────────────────────────────────
banner "3. Tests, docs, benches"
step "test debug (workspace)"         600 -- cargo test --workspace --all-features
if [ $QUICK -eq 0 ]; then
  step "test release (workspace)" 600 -- cargo test --workspace --release --all-features
fi
step "doc build (--no-deps)"          300 -- env "RUSTDOCFLAGS=-D warnings" cargo doc --no-deps --workspace --all-features
# `pyo3/extension-module` est réservé à Maturin ; Cargo peut valider tout le workspace.
step "benches compile (workspace)"    600 -- cargo bench --workspace --all-features --no-run

# ── 4. cross-compile (aarch64 NEON path) ────────────────────────────────────
banner "4. Cross-compile (aarch64 / NEON path)"
if [ $DO_CROSS -eq 1 ] && rustup target list --installed 2>/dev/null | grep -q '^aarch64-unknown-linux-gnu'; then
  step "cross-build lib (aarch64)"    300 -- cargo build -p scirust --lib --target aarch64-unknown-linux-gnu
else
  skip "cross-build lib (aarch64)" "run: rustup target add aarch64-unknown-linux-gnu"
fi

# ── 5. exercise every example ───────────────────────────────────────────────
banner "5. Exercise every example (release)"
# Examples are perf demonstrations; always run them in release so they are fast
# and representative — debug-mode SGD/throughput loops are pointlessly slow and
# can hit the per-step timeout. (--quick only skips the separate release
# workspace build/test in sections 2–3, not the examples.)
BUILD_FLAG="--release"
export BUILD_FLAG
shopt -s nullglob
for f in scirust/examples/*.rs; do
  ex="$(basename "$f" .rs)"
  step "example: $ex" 300 -- cargo run -q $BUILD_FLAG --example "$ex"
done
shopt -u nullglob

# ── 5b. self-audit (runs every internal invariant; exit != 0 if any fails) ──
banner "5b. Self-audit (slha-audit)"
step "slha-audit --json" 180 -- cargo run -q $BUILD_FLAG --bin slha-audit -- --json

# ── 6. determinism (reproducibility claim) ──────────────────────────────────
banner "6. Determinism (same input ⇒ same output)"
step "determinism: basic_usage ×2" 180 -- bash -c '
  a=$(cargo run -q $BUILD_FLAG --example basic_usage 2>&1)
  b=$(cargo run -q $BUILD_FLAG --example basic_usage 2>&1)
  if [ "$a" != "$b" ]; then echo "OUTPUT DIFFERS:"; diff <(printf "%s" "$a") <(printf "%s" "$b"); exit 1; fi'

# ── 7. soak (optional) ──────────────────────────────────────────────────────
if [ "$SOAK" -gt 0 ]; then
  banner "7. Soak (${SOAK}× hot kernel)"
  step "soak: measure ×$SOAK" 1800 -- bash -c 'for i in $(seq 1 '"$SOAK"'); do cargo run -q --release --example measure >/dev/null 2>&1 || { echo "iteration $i failed"; exit 1; }; done'
fi

# ── 8. optional deep checks (informational) ─────────────────────────────────
banner "8. Optional deep checks"
# Miri can't execute target_feature SIMD intrinsics (AVX/NEON), so a full
# `cargo miri test` would error on the SIMD-equivalence tests — documented skip.
skip "miri (UB on unsafe code)" "manual: rustup +nightly component add miri (note: cannot run SIMD intrinsics)"
if have cargo-audit; then
  step "cargo audit (advisories)" 180 -- cargo audit
else
  skip "cargo audit" "zero runtime deps ⇒ low value; install: cargo install cargo-audit"
fi

# ── reports ─────────────────────────────────────────────────────────────────
{
  echo "# SLHA v2 — Stress Report"
  echo
  echo "- **When:** \`$STAMP\`"
  echo "- **Host:** \`$(uname -srm)\`"
  echo "- **rustc:** \`$(rustc --version 2>/dev/null)\`"
  echo "- **Mode:** $([ $QUICK -eq 1 ] && echo quick || echo full) (soak=$SOAK, cross=$DO_CROSS)"
  echo "- **Result:** ✅ $PASS passed · ❌ $FAIL failed · ⏭️ $SKIP skipped"
  echo
  echo "| Status | Step | Time | Note |"
  echo "|:--:|---|--:|---|"
  for row in "${ROWS[@]}"; do IFS='|' read -r st nm tm nt <<<"$row"; echo "| $st | $nm | $tm | $nt |"; done
  echo
  if [ $FAIL -gt 0 ]; then
    echo "## Failures"
    for n in "${FAILED_NAMES[@]}"; do echo "- \`$n\` — see \`logs-$STAMP/\`"; done
  fi
  echo
  echo "_Per-step logs: \`target/stress/logs-$STAMP/\`_"
} >"$REPORT_MD"

printf '{\n  "timestamp": "%s",\n  "host": "%s",\n  "rustc": "%s",\n  "passed": %d,\n  "failed": %d,\n  "skipped": %d,\n  "ok": %s\n}\n' \
  "$STAMP" "$(uname -srm)" "$(rustc --version 2>/dev/null | sed 's/"/\\"/g')" "$PASS" "$FAIL" "$SKIP" "$([ $FAIL -eq 0 ] && echo true || echo false)" \
  >"$SUMMARY_JSON"

# ── summary ─────────────────────────────────────────────────────────────────
banner "Summary"
echo "  ${G}PASS:$PASS${N}   ${R}FAIL:$FAIL${N}   ${Y}SKIP:$SKIP${N}"
echo "  report: $REPORT_MD"
echo "  json:   $SUMMARY_JSON"
echo "  logs:   $LOG_DIR/"
if [ $FAIL -eq 0 ]; then echo "  ${G}${B}ALL GREEN${N}"; exit 0; else echo "  ${R}${B}FAILURES PRESENT${N}"; exit 1; fi
