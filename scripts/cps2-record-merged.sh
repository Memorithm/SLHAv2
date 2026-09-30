#!/usr/bin/env bash
# Record only a confirmed CPS-2 merge in the existing off-main agent roadmap.
set -euo pipefail
umask 077
export GIT_TERMINAL_PROMPT=0
REPO=Memorithm/SLHAv2
HEAD=1462533ebf43db0533843283f958d265b262ae02
BRANCH=agent/ecosystem-roadmap
FILE=.agent/SLHAV2_ECOSYSTEM_ROADMAP.yaml
WORK=$(mktemp -d "${RUNNER_TEMP:?}/cps2-roadmap.XXXXXXXX")
gh api "repos/$REPO/pulls/131" > "$WORK/pr.json"
jq -e --arg head "$HEAD" '.merged == true and .head.sha == $head and .base.ref == "master"' "$WORK/pr.json" >/dev/null
MERGE=$(jq -r .merge_commit_sha "$WORK/pr.json")
[[ "$MERGE" =~ ^[0-9a-f]{40}$ ]]
SOURCE=$(gh api "repos/$REPO/git/ref/heads/$BRANCH" --jq '.object.sha')
[[ "$SOURCE" =~ ^[0-9a-f]{40}$ ]]
mkdir "$WORK/source"
git -C "$WORK/source" init -q
git -C "$WORK/source" remote add origin "https://github.com/$REPO.git"
timeout 90 git -C "$WORK/source" fetch --quiet --depth=1 origin "$SOURCE"
git -C "$WORK/source" checkout --quiet --detach FETCH_HEAD
cd "$WORK/source"
test "$(git rev-parse HEAD)" = "$SOURCE"
test "$(git hash-object "$FILE")" = 9ac5afb72be66ebfc35a5cd36b819e3c1e23758a
! grep -q '^cps2_destination_verification_2026_09_30:' "$FILE"
sed -i 's/^last_updated: .*/last_updated: "2026-09-30"/' "$FILE"
cat >> "$FILE" <<EOF

cps2_destination_verification_2026_09_30:
  status: merged_synthetic_diagnostic_only
  pull_request: 131
  qualified_head: "$HEAD"
  merge_revision: "$MERGE"
  kvlab_producer: "265a2a65120b8f6f06c3cbb6604cf821ceda94d0"
  flat_producer: "ad1634fc922f6223dd3a83ac84154a82b1a35562"
  schema: kvlab.cps2-compact-quality/v1
  raw_csv_sha256: "2457b70e8e75cf29db366f8b11761efa084f1da3bec9ed49133c6436ac2c7216"
  public_cpu_worker_run: 36773552599
  checksum_gated_import_run: 36774007835
  complete_synthetic_observations: 50
  destination_regression_tests: 14
  implementation: scirust/src/kvlab_cps2.rs
  protocol: docs/CPS2_IMPORT.md
  execution_evidence: docs/CPS2_IMPORT_EVIDENCE.md
  invariants:
    - read_only_diagnostic_not_a_runtime_selector
    - revision_labels_are_assertions_not_authenticated_worker_attestations
    - adverse_dominant_key_fixture_is_retained_as_negative_evidence
    - no_LR1_candidate_protected_holdout_scorer_codec_or_cache_default_change
    - no_model_quality_memory_traffic_latency_or_GPU_promotion
    - ML_maturity_and_existing_real_model_quality_gates_remain_unchanged
EOF
git diff --check
git add -- "$FILE"
test "$(git diff --cached --name-only | wc -l)" -eq 1
test "$(timeout 30 git ls-remote origin "refs/heads/$BRANCH" | cut -f1)" = "$SOURCE"
git -c user.name='Memorithm RemoteOps' -c user.email='remoteops@memorithm.local' commit -m 'docs(agent): record merged CPS-2 destination diagnostic verification'
FINAL=$(git rev-parse HEAD)
timeout 60 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin "HEAD:refs/heads/$BRANCH"
printf 'CPS2_ROADMAP_RECEIPT_COMMITTED head=%s merge=%s\n' "$FINAL" "$MERGE"
