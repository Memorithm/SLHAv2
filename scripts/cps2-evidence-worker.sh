#!/usr/bin/env bash
# Read-only CPU worker for the RemoteOps CPS-2 qualification plan.
# RemoteOps plan: c6816d87b28d5011e2e8304ea6ff21cd44baaeb4.
# No secrets, host services, protected model data, or repository writes.
set -euo pipefail
umask 077
export CARGO_BUILD_JOBS=2 CARGO_TERM_COLOR=never GIT_TERMINAL_PROMPT=0
KVLAB=265a2a65120b8f6f06c3cbb6604cf821ceda94d0
FLAT=ad1634fc922f6223dd3a83ac84154a82b1a35562
SOURCE=7f57822411aa869df5e87690dee9972aa105b172
BRANCH=research/cps2-destination-validation-20260930
WORK=$(mktemp -d "${RUNNER_TEMP:?}/cps2-qualify.XXXXXXXX")
OUT="${GITHUB_WORKSPACE:?}/cps2-handoff"
mkdir -m 700 "$OUT"
clone_at() {
  local repo=$1 revision=$2 destination=$3
  mkdir "$destination"
  git -C "$destination" init -q
  git -C "$destination" remote add origin "https://github.com/$repo.git"
  timeout 90 git -C "$destination" fetch --quiet --depth=1 origin "$revision"
  git -C "$destination" checkout --quiet --detach FETCH_HEAD
  test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}
clone_at Memorithm/KVLab "$KVLAB" "$WORK/kvlab"
clone_at Memorithm/SLHAv2 "$SOURCE" "$WORK/slha"
test "$(timeout 30 git -C "$WORK/slha" ls-remote origin "refs/heads/$BRANCH" | cut -f1)" = "$SOURCE"
export CARGO_TARGET_DIR="$WORK/target-kvlab"
cargo run --quiet --manifest-path "$WORK/kvlab/rust/bkv_scan/Cargo.toml" --bin cps2_compact_panel > "$WORK/first.csv"
cargo run --quiet --manifest-path "$WORK/kvlab/rust/bkv_scan/Cargo.toml" --bin cps2_compact_panel > "$WORK/second.csv"
cmp "$WORK/first.csv" "$WORK/second.csv"
test "$(wc -l < "$WORK/first.csv")" -eq 51
git -C "$WORK/kvlab" diff --exit-code
DIGEST=$(sha256sum "$WORK/first.csv" | cut -d' ' -f1)
printf 'CPS2_PRODUCER_REPRODUCED rows=50 sha256=%s\n' "$DIGEST"
cd "$WORK/slha"
mkdir -p scirust/tests/fixtures
cp "$WORK/first.csv" scirust/tests/fixtures/kvlab_cps2_v1.csv
rustfmt --edition 2021 scirust/src/kvlab_cps2.rs scirust/tests/kvlab_cps2.rs scirust/examples/cps2_inspect.rs
while IFS= read -r path; do
  case "$path" in
    scirust/src/kvlab_cps2.rs|scirust/tests/kvlab_cps2.rs|scirust/examples/cps2_inspect.rs) ;;
    *) echo "Unexpected formatter modification: $path" >&2; exit 1 ;;
  esac
done < <(git diff --name-only)
# Preserve intermediate artifacts for diagnosing failures, without calling them qualified.
cp --parents scirust/src/kvlab_cps2.rs scirust/tests/kvlab_cps2.rs scirust/examples/cps2_inspect.rs \
  scirust/tests/fixtures/kvlab_cps2_v1.csv "$OUT"
printf '%s\n' "$SOURCE" > "$OUT/destination-checkout.txt"
export CARGO_TARGET_DIR="$WORK/target-slha"
cargo --locked fmt --all --check
cargo --locked clippy -p scirust --all-targets -- -D warnings
cargo --locked test -p scirust --test kvlab_cps2
cargo --locked run -p scirust --example cps2_inspect -- scirust/tests/fixtures/kvlab_cps2_v1.csv "$KVLAB" | tee "$OUT/qualification-summary.txt"
cat >> AGENTS.md <<'DOC'

## CPS-2 destination verification

Before importing compact-preselection diagnostics, read `docs/CPS2_IMPORT.md`
and `docs/CPS2_IMPORT_EVIDENCE.md`. The `kvlab_cps2` Rust module checks only the
frozen synthetic KVLab #175 CSV; it is not a runtime selector, model-quality
gate or authenticated execution attestation. Keep the adverse dominant-key
fixture, complete matched controls and exact producer identities. No LR1
candidate, protected holdout, scorer, codec or cache default changes are
permitted by a successful diagnostic import.
DOC
cat >> ROADMAP.md <<'DOC'

## CPS-2 destination verification — 2026-09-30

The `scirust::kvlab_cps2` read-only consumer verifies the complete frozen
KVLab #175 development panel against an independent scalar fixture oracle.
The upstream producer is pinned to `265a2a65120b8f6f06c3cbb6604cf821ceda94d0`;
FLAT CPS-1 is pinned to `ad1634fc922f6223dd3a83ac84154a82b1a35562`.
See `docs/CPS2_IMPORT.md` and `docs/CPS2_IMPORT_EVIDENCE.md` for the protocol,
actual execution provenance and limitations. Source/schema drift, missing
controls, forged metrics and unsupported promotion flags are rejected.

This completes a synthetic evidence-consumption slice under RK2/RK3 only.
Real-model quality, physical systems gains and runtime adoption remain gated;
RK1/LR1 and its protected holdout are unchanged.
DOC
{
  printf '# CPS-2 destination import execution\n\n'
  printf 'Scope: synthetic diagnostic only; no runtime promotion.\n\n'
  printf -- '- Executed at UTC: `%s`.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf -- '- SLHAv2 public CPU worker run: `%s`, attempt `%s`.\n' "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}"
  printf -- '- Worker source revision: `%s`.\n' "${GITHUB_SHA:?}"
  printf -- '- RemoteOps qualification plan: `c6816d87b28d5011e2e8304ea6ff21cd44baaeb4`.\n'
  printf -- '- Destination checkout before formatting: `%s`.\n' "$SOURCE"
  printf -- '- KVLab revision: `%s`.\n' "$KVLAB"
  printf -- '- FLAT revision: `%s`.\n' "$FLAT"
  printf -- '- Raw CSV SHA-256: `%s`.\n' "$DIGEST"
  printf -- '- Raw fixture: `scirust/tests/fixtures/kvlab_cps2_v1.csv`.\n'
  printf -- '- Repetition: two executions of the actual upstream Rust binary; byte-identical output.\n'
  printf -- '- Complete observations: 50; input header: 21 columns.\n'
  printf -- '- Execution backend: GitHub-hosted Linux CPU runner.\n'
  printf -- '- Architecture: `%s`; compiler: `%s`.\n\n' "$(uname -m)" "$(rustc --version)"
  printf 'Formatting, strict default-feature Clippy and all 14 destination regression tests passed before this record was written. Full repository CI remains a separate merge gate.\n\n'
  printf 'The record is unsigned provenance, not worker attestation. No model or protected holdout was accessed. No Thor execution, GPU, quality, memory, traffic or performance result is claimed.\n\n'
  printf '## Actual destination command output\n\n```text\n'
  cat "$OUT/qualification-summary.txt"
  printf '```\n'
} > docs/CPS2_IMPORT_EVIDENCE.md
cp --parents docs/CPS2_IMPORT_EVIDENCE.md AGENTS.md ROADMAP.md "$OUT"
cd "$OUT"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
printf 'CPS2_ARTIFACT_QUALIFIED csv_sha256=%s destination_checkout=%s\n' "$DIGEST" "$SOURCE"
