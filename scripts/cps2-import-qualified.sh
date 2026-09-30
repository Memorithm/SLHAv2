#!/usr/bin/env bash
# Explicit write stage, separate from the read-only scientific worker.
# Only the predeclared destination branch can be advanced; never force-push.
set -euo pipefail
umask 077
export CARGO_BUILD_JOBS=2 CARGO_TERM_COLOR=never GIT_TERMINAL_PROMPT=0
REPO=Memorithm/SLHAv2
SOURCE=7f57822411aa869df5e87690dee9972aa105b172
BRANCH=research/cps2-destination-validation-20260930
RUN=36773552599
ARTIFACT=11124875209
WORKER=d8890c3d521df23746be52b641db442a4d2081d9
ZIP_SHA=21a43de50f3392399002bd77aff2a38c800c56511527d42322a0bd0cf1fba9fc
CSV_SHA=2457b70e8e75cf29db366f8b11761efa084f1da3bec9ed49133c6436ac2c7216
WORK=$(mktemp -d "${RUNNER_TEMP:?}/cps2-import.XXXXXXXX")
# GitHub API provenance and exact immutable artifact bytes are both checked.
gh api "repos/$REPO/actions/runs/$RUN" > "$WORK/run.json"
jq -e --arg sha "$WORKER" '.head_sha == $sha and .status == "completed" and .conclusion == "success" and .path == ".github/workflows/cps2-evidence-worker.yml"' "$WORK/run.json" >/dev/null
gh api "repos/$REPO/actions/artifacts/$ARTIFACT" > "$WORK/artifact.json"
jq -e --arg sha "$WORKER" --argjson run "$RUN" '.expired == false and .workflow_run.id == $run and .workflow_run.head_sha == $sha' "$WORK/artifact.json" >/dev/null
gh api "repos/$REPO/actions/artifacts/$ARTIFACT/zip" > "$WORK/handoff.zip"
printf '%s  %s\n' "$ZIP_SHA" "$WORK/handoff.zip" | sha256sum --check
mkdir "$WORK/handoff"
unzip -q "$WORK/handoff.zip" -d "$WORK/handoff"
(cd "$WORK/handoff" && sha256sum --check SHA256SUMS)
test "$(cat "$WORK/handoff/destination-checkout.txt")" = "$SOURCE"
printf '%s  %s\n' "$CSV_SHA" "$WORK/handoff/scirust/tests/fixtures/kvlab_cps2_v1.csv" | sha256sum --check
mkdir "$WORK/source"
git -C "$WORK/source" init -q
git -C "$WORK/source" remote add origin "https://github.com/$REPO.git"
timeout 90 git -C "$WORK/source" fetch --quiet --depth=1 origin "$SOURCE"
git -C "$WORK/source" checkout --quiet --detach FETCH_HEAD
cd "$WORK/source"
test "$(git rev-parse HEAD)" = "$SOURCE"
test "$(timeout 30 git ls-remote origin "refs/heads/$BRANCH" | cut -f1)" = "$SOURCE"
files=(
  scirust/src/kvlab_cps2.rs
  scirust/tests/kvlab_cps2.rs
  scirust/examples/cps2_inspect.rs
  scirust/tests/fixtures/kvlab_cps2_v1.csv
  docs/CPS2_IMPORT_EVIDENCE.md
  AGENTS.md
  ROADMAP.md
)
for path in "${files[@]}"; do
  test -f "$WORK/handoff/$path"
  test ! -L "$WORK/handoff/$path"
  mkdir -p "$(dirname "$path")"
  cp "$WORK/handoff/$path" "$path"
  cmp "$WORK/handoff/$path" "$path"
done
export CARGO_TARGET_DIR="$WORK/target"
cargo --locked fmt --all --check
cargo --locked clippy -p scirust --all-targets -- -D warnings
cargo --locked test -p scirust --test kvlab_cps2
cargo --locked run -p scirust --example cps2_inspect -- scirust/tests/fixtures/kvlab_cps2_v1.csv 265a2a65120b8f6f06c3cbb6604cf821ceda94d0 > "$WORK/summary.txt"
cmp "$WORK/handoff/qualification-summary.txt" "$WORK/summary.txt"
git add -- "${files[@]}"
test "$(git diff --cached --name-only | wc -l)" -eq 7
test "$(timeout 30 git ls-remote origin "refs/heads/$BRANCH" | cut -f1)" = "$SOURCE"
git -c user.name='Memorithm RemoteOps' -c user.email='remoteops@memorithm.local' \
  commit -m 'test(cps2): retain independently verified upstream CSV and destination evidence'
FINAL=$(git rev-parse HEAD)
timeout 60 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push origin "HEAD:refs/heads/$BRANCH"
test "$(timeout 30 git ls-remote origin "refs/heads/$BRANCH" | cut -f1)" = "$FINAL"
printf 'CPS2_QUALIFIED_IMPORT_COMMITTED head=%s raw_csv_sha256=%s source_worker=%s\n' "$FINAL" "$CSV_SHA" "$WORKER"
