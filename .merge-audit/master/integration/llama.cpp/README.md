# SLHA v2 × llama.cpp — K-cache round-trip and compressed-score quality gates

This directory implements and measures real-LLM quality-path integrations for
SLHA v2 on llama.cpp. Two milestones are recorded here:

1. **K-cache round-trip** — every K vector is encoded to a 128-byte SLHA tile
   and decoded back to the original K dimension *before* it is stored in
   llama.cpp's normal KV cache. Attention is untouched.
2. **Direct compressed-score replacement path** — a custom GGML operation
   replaces the attention logits with scores computed directly from the
   compressed SLHA tiles, after llama.cpp has already materialised the baseline
   Q·K product.

> Neither milestone is a fused attention kernel or a physically compressed
> KV-cache implementation. No KV-cache memory reduction and no attention-speed
> gain is claimed.

## Status

Implemented and measured:

* Inert passthrough hook (proves the custom GGML op does not move perplexity).
* Per-layer K activation collection mode (writes `layer-N-k.bin`).
* Automated per-layer projection training script (outputs `layer-NNN.slhw` +
  `manifest.json`).
* SLHA K round-trip callback using `slha_encode_key` + `slha_decode_key`.
* Shadow-score quality gate: `SLHA_SCORE_MODE=shadow` compares baseline Q·K
  logits with direct SLHA scores while leaving attention output unchanged.
* **Strict direct compressed-score replacement**: `SLHA_SCORE_MODE=replace`
  substitutes the logits and fails closed unless every active vector was
  replaced.
* Reproducible build/apply/measure scripts.

## The direct compressed-score replacement path

What it **does**:

* A custom GGML operation is inserted after `build_attn_mha` computes `kq`.
  It overwrites the attention logits with scores computed directly from the
  compressed SLHA tile side-store.
* It is **fail-closed**: the run is rejected unless every active vector was
  replaced, with no failures and no fallbacks.
* Active logits and padding logits are accounted for **separately**. Padding
  and inactive-stream positions are excluded from the count of logits directly
  computed by SLHA.

What it **does not** do:

* **Baseline Q·K is still materialised by llama.cpp.** The custom operation
  replaces logits *after* the baseline matrix multiplication; it does not avoid
  that matrix multiplication.
* No fused attention kernel is implemented.
* No physical K-cache memory reduction is claimed. The compressed tiles live in
  a side-store; the normal KV cache is unchanged.
* Strict replace mode currently supports **one parallel sequence** only
  (`n_stream == 1`).

## Provenance of the recorded measurements

```text
SLHAv2 implementation commit : 6361dfdbcd30660bf2d623fe19029938dd209cd7
llama.cpp commit             : fdb1db877c526ec90f668eca1b858da5dba85560 (tag b9860)
measurement branch           : claude/compressed-score-quality-gate-qiqbm5
```

The original feature-branch commits (canonical compressed-score C API, shadow
score gate, richer shadow metrics, strict fail-closed replacement) were
**squash-merged through PR #57** into implementation commit `6361dfd`. The
strict replacement implementation was therefore measured from `6361dfd`; the
pre-merge SHAs no longer exist in this repository. The documentation commit
recording these results is necessarily later than the implementation commit it
measures.

## Headline results

Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 test, mixed codec. All runs used
`chunks = 12`, `context length = 512`, `batch size = 512`,
`parallel sequences = 1`, `threads = 4`, `flash attention = off`, verified from
each run's own runtime output.

| Metric | Value |
| --- | ---: |
| Unpatched baseline mean PPL (rung A, n=3) | 11.8779 |
| Pass-through control mean PPL (rung D, n=3) | 11.8831 |
| Strict replace mean PPL (rung F, n=3) | 16.9173 |
| Strict replace sample standard deviation | 0.0525 |
| Primary gap vs pass-through control (F − D) | +5.0342 PPL |
| Primary relative gap | +42.364 % |
| Active coverage | 1 |
| Failed vectors | 0 |
| Fallback vectors | 0 |
| Padding logits audited | 170688 |
| Padding nonzero count | 0 |
| Strict replace seconds per pass | ≈ 9.40 |
| Strict replace tokens per second | ≈ 54.5 |
| Pass-through control seconds per pass | ≈ 3.02 |
| Pass-through control tokens per second | ≈ 169.5 |

The direct compressed-score replacement path produced a large observed PPL
degradation relative to the pass-through integration control. The effect was
far larger than the run-to-run variability observed in the control ladder. Its
precise cause has not yet been isolated.

The observed quality gap applies to the current model, mixed codec, projection
weights, score implementation and evaluation protocol. It is **not** attributed
solely to quantization. The experiment does not determine whether the gap
originates from codec quantization, projection training, score scaling, query
preparation, attention temperature, layer-specific error accumulation, padding
behaviour, tile representation or another integration effect.

## Six-rung control ladder

Three complete runs per rung, identical protocol. All 18 accepted runs exited
with status `0`.

| Rung | Description | run 1 | run 2 | run 3 | mean | sample sd | spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | unpatched baseline | 11.8837 | 11.8737 | 11.8764 | 11.8779 | 0.0052 | 0.0100 |
| B | patched llama.cpp, SLHA fully off | 11.8770 | 11.8766 | 11.8873 | 11.8803 | 0.0061 | 0.0107 |
| C | tilestore K-write hook only | 11.8785 | 11.8829 | 11.8787 | 11.8800 | 0.0025 | 0.0044 |
| D | padaudit pass-through custom operation | 11.8775 | 11.8768 | 11.8949 | 11.8831 | 0.0103 | 0.0181 |
| E | padzero | 11.8811 | 11.8872 | 11.8762 | 11.8815 | 0.0055 | 0.0110 |
| F | strict compressed-score replacement | 16.8573 | 16.9396 | 16.9549 | 16.9173 | 0.0525 | 0.0976 |

The **observed unpatched-baseline run-to-run spread = 0.0100 PPL across three
runs**. This is an observed variability figure, not a formally characterised
numerical noise floor.

### Contrasts

Difference of means with propagated standard error. All intervals are
**approximate exploratory intervals based on three independent runs per group**
(t = 2.776, df = 4). **Statistical power is limited.**

| Contrast | Δ PPL | Δ % | approximate 95 % CI | Interpretation |
| --- | ---: | ---: | --- | --- |
| B − A, patch/build | +0.0024 | +0.020 % | [−0.0104, +0.0151] | not resolved at this sample size |
| C − B, tilestore hook | −0.0003 | −0.002 % | [−0.0108, +0.0102] | not resolved at this sample size |
| D − C, pass-through custom operation | +0.0030 | +0.026 % | [−0.0139, +0.0199] | not resolved at this sample size |
| E − D, padding zeroing | −0.0016 | −0.013 % | [−0.0202, +0.0171] | not resolved at this sample size |
| F − D, direct score substitution | +5.0342 | +42.364 % | [+4.9485, +5.1199] | large resolved effect |
| F − A, secondary comparison | +5.0393 | +42.426 % | [+4.9548, +5.1239] | large resolved effect |

"Not resolved at this sample size" is **not** a claim of equivalence or of a
proven absent effect.

Numerical PPL results show run-to-run variation even in **unpatched**
llama.cpp. The production counters were fully deterministic across repeats; the
numerical variation was not instrumentally isolated.

## Strict replacement counters

Identical across all three accepted runs, imported programmatically from the
immutable run logs:

```text
callbacks                = 1456
active_expected_vectors  = 2065056
active_replaced_vectors  = 2065056
active_expected_logits   = 1056965952
active_replaced_logits   = 1056965952
padding_vectors          = 672
padding_logits           = 170688
inactive_stream_vectors  = 0
inactive_stream_logits   = 0
failed_vectors           = 0
fallback_vectors         = 0
missing_tile             = 0
query_prep_fail          = 0
score_fail               = 0
nonfinite_score          = 0
unsupported_shape        = 0
unsupported_stride       = 0
error_code               = 0
n_stream                 = 1
active_coverage          = 1
valid                    = true
```

`build_and_roundtrip.sh replace` rejects the run unless these conditions hold,
so incomplete or fallback runs are never reported as results.

## Padding audit

Deterministic raw-logit audit over the padded region `k >= n_written`,
identical across all six diagnostic runs:

```text
audited_padded_vectors           = 672
audited_padded_logits            = 170688
max_abs_padded_baseline_logit    = 0
nonzero_padded_baseline_logits   = 0
nonfinite_padded_baseline_logits = 0
```

`audited_padded_logits` cross-checks exactly against the strict-replacement
`padding_logits` counter.

For this pinned model and protocol, all audited padded baseline Q·K logits were
exactly zero. Writing `0.0f` to those positions is therefore an arithmetic
no-op.

The empirical `padzero − padaudit` comparison (−0.0016 PPL, approximate 95 % CI
[−0.0202, +0.0171], not resolved at this sample size) is **supporting
evidence, not the primary proof**.

This exact-zero result must **not** be generalised to other models, other
llama.cpp versions or other KV-cache implementations.

## Throughput

| Path | seconds per pass | tokens per second |
| --- | ---: | ---: |
| padaudit pass-through control | ≈ 3.02 | ≈ 169.5 |
| strict compressed-score replacement | ≈ 9.40 | ≈ 54.5 |

Strict replacement is approximately **3.11× slower by seconds per pass**. This
is a quality-gate implementation using a custom CPU operation — not a fused
kernel and not an optimised performance benchmark.

## Calibration preprocessing (out-of-tree)

The projection weights used by the strict replacement runs were **not** trained
directly from the raw collected dumps. A deterministic out-of-tree row-removal
step was required first:

```text
27 total rows removed
layers 1–27 affected
one removed row per affected layer
row index 0 or 1
all removed rows contained NaN
zero removed rows contained positive or negative infinity
6146 raw rows to 6145 clean rows per affected layer
layer 0 unchanged
```

The rule is deterministic and limited to removing complete rows containing any
non-finite value. No finite value was imputed, replaced, clamped or otherwise
modified.

```text
sanitization script SHA-256:
4f0d7e3f0bbd557afd99085ca899d45bc4916bd0a9e4d014e26f7775b9ba32a4
```

The position and pattern are consistent with a warmup-like callback, but the
exact upstream origin was not instrumentally proven.

The weights were generated after the squash merge from implementation commit
`6361dfdbcd30660bf2d623fe19029938dd209cd7` using an out-of-tree deterministic
non-finite-row removal step.

Per-layer raw and cleaned hashes, aggregate calibration hashes, all weight-file
hashes, the manifest hash and the exact training command are recorded in
[`results/measurements.json`](results/measurements.json).

> **Update.** The out-of-tree sanitization above is no longer required: the
> production pipeline now validates calibration and fails closed on any
> non-finite row. See **Calibration integrity policy** below.

## Calibration integrity policy

The collection and training pipeline is fail-safe with respect to non-finite
(NaN / +Inf / -Inf) K-activation rows. A single shared validator
(`shim/slha_calibration.{hpp,cpp}`) is linked by the collection driver, the
trainer gate CLI (`shim/slha_calibrate_cli.cpp`) and the unit tests
(`tests/calibration_tests.cpp`), so all three enforce identical rules.

**Default policy — `reject`.** Any non-finite scalar in any dump makes the whole
calibration invalid and fails the run with a non-zero status. Structural
defects — empty files, truncated payloads, a size that disagrees with the
header, dimensions that differ across layers, missing layers (when the expected
count is known), or duplicate layer ids — fail the same way. Nothing is
silently removed while still returning success.

**Manifest.** `build_and_roundtrip.sh collect` writes
`<calib_dir>/calibration_manifest.json` recording: format version,
implementation and llama.cpp commits, model identifier and hash, dataset hash,
collection command, UTC timestamp, per-layer rows observed / accepted /
rejected, NaN / +Inf / -Inf row counts, per-layer non-finite row indices,
per-layer raw and clean SHA-256, `sanitized`, and the global `valid` flag. A
valid collection has `total_rows_rejected == 0`, `nan_row_count == 0`,
`sanitized == false`, `valid == true`.

**Fail before training.** `train_layer_weights.sh` runs the same validator as a
gate *before* fitting any projection. On failure it exits non-zero and writes no
weights. Training then proceeds into a staging directory; only after every
expected `.slhw` exists and is non-empty, and the manifest is written, is the
staging directory moved into the final destination. On any failure the staging
directory is removed and a pre-existing valid weights directory is left
untouched (atomic swap). The trainer also cross-checks a collection manifest
when present and rejects a `valid=false` manifest or a codec / dimension /
model / dataset mismatch.

**Research/recovery mode — `drop-row` (never the default).** Set
`SLHA_CALIBRATION_NONFINITE_POLICY=drop-row` to have the validator remove whole
rows that contain any non-finite scalar. Finite values are never clamped or
imputed; removed rows are recorded; the output is marked `sanitized=true`; raw
and clean hashes are both recorded; and the result is accepted only when each
layer retains at least `SLHA_CALIBRATION_MIN_ROWS` rows (default 1). This
reproduces the earlier out-of-tree cleanup, but explicitly and inside the
production pipeline.

```bash
# Production default (fails closed on any non-finite row):
CALIB_DIR=/tmp/slha-llama/calibration \
  DATA_FILE=/tmp/slha-llama/wiki.train.raw \
  WORK=/tmp/slha-llama \
  integration/llama.cpp/build_and_roundtrip.sh collect

# Research/recovery only (deterministic whole-row removal):
SLHA_CALIBRATION_NONFINITE_POLICY=drop-row \
  CALIB_DIR=/tmp/slha-llama/calibration \
  WEIGHTS_DIR=/tmp/slha-llama/weights \
  WORK=/tmp/slha-llama \
  integration/llama.cpp/scripts/train_layer_weights.sh mixed
```

**Origin (observed vs inferred).** Instrumentation of the collect callback
showed that the first collect invocation for every layer processes exactly two
tokens at absolute rows 0–1, before the twelve 512-token evaluation passes
(`n_tokens=2` at `row_base=0`, then `n_tokens=512`). The earlier non-finite
rows sat at exactly those indices (0 or 1). In the earlier session each affected
layer had one full 256-wide row of NaN (27 layers × 256 = 6912 non-finite
scalars), while a fresh collection on a different host produced zero non-finite
rows from the identical pipeline. *Observed:* the corrupt rows are whole-row and
originate in the initial two-token priming decode, and the defect is
intermittent across hosts. *Inferred:* the priming decode reads a K slot that
was not populated with valid activations, so its contents are host-dependent
garbage. *Not proven:* the precise upstream mechanism inside llama.cpp. Because
the trigger is intermittent and only distinguishable by symptoms (a low row
index, or the value being non-finite), the robust fix is the validation
boundary above rather than a source-side exclusion predicate.

## Excluded runs

The following runs are recorded for audit but **none of their results are
included in any aggregate**:

| Excluded run | Reason |
| --- | --- |
| strict replace against incomplete weights | training aborted on non-finite rows; only 1 of 28 weight files existed and no `manifest.json` had been written |
| first padaudit/padzero builds | build directory copied with `cp -a` retained the wrong `CMAKE_HOME_DIRECTORY`, so the diagnostic sources were never recompiled |
| first padaudit/padzero binaries | both supposedly distinct variants hashed identically, proving the variant sources had not been compiled in |
| padaudit run under those binaries | log shows `unknown SLHA_SCORE_MODE='paddiag', falling back to off`, so it measured score-mode-off |
| combined environment-selected diagnostic binary | one binary selecting both diagnostic modes by environment variable; superseded by dedicated per-variant builds with distinct binary hashes |

## Why Qwen2.5-1.5B instead of 0.5B

The integration seam is `llama_kv_cache::cpy_k`, where K has shape
`[n_embd_gqa = head_dim × n_kv_heads]`. Qwen2.5-0.5B uses GQA with
`n_kv_heads = 2` and `head_dim = 64`, so `n_embd_gqa = 128`. SLHA requires
`d > D_C = 128` for a non-trivial residual, so the 0.5B model cannot be used at
this seam. The 1.5B checkpoint has `n_embd_gqa = 256`, satisfying the constraint.

## Layout

```text
integration/llama.cpp/
├── README.md                         # this file
├── build_and_roundtrip.sh            # clone, patch, build, run all modes
├── patches/0001-slha-k-passthrough.patch
├── shim/slha_llama.cpp               # C++ shim (collect / passthrough / roundtrip / tilestore / shadow / replace)
├── shim/slha_llama.hpp
├── shim/slha_replace_counters.cpp    # strict replacement counters
├── shim/slha_replace_counters.hpp
├── shim/slha_calibration.cpp         # calibration integrity validator (shared core)
├── shim/slha_calibration.hpp
├── shim/slha_calibrate_cli.cpp       # slha_calibrate — validation/manifest gate
├── tests/replace_strict_tests.cpp    # production-linked strict-counter tests
├── tests/calibration_tests.cpp       # production-linked calibration validator tests
├── tests/trainer_atomicity_test.sh   # trainer fail-before-training / atomic-swap tests
├── scripts/prepare_calibration.sh    # build a separate calibration corpus
├── scripts/train_layer_weights.sh    # validate calibration, then atomically train .slhw
└── results/
    ├── measurements.json             # machine-readable results and provenance
    └── README.md                     # Markdown report
```

## Quick reproduction

```bash
# 1. Build the C bridge
cargo build --release -p slha-c

# 2. Baseline
WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh baseline

# 3. Passthrough sanity
WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh passthrough

# 4. Calibration corpus (must be separate from evaluation data)
WORK=/tmp/slha-llama integration/llama.cpp/scripts/prepare_calibration.sh

# 5. Collect K activations on the calibration corpus
CALIB_DIR=/tmp/slha-llama/calibration \
  DATA_FILE=/tmp/slha-llama/wiki.train.raw \
  WORK=/tmp/slha-llama \
  integration/llama.cpp/build_and_roundtrip.sh collect

# 6. Train per-layer projections (mixed codec). The trainer validates the
#    calibration first (default policy 'reject'): a non-finite row aborts the
#    run with no weights written. See "Calibration integrity policy".
MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  MODEL_FILE=qwen2.5-1.5b-instruct-q8_0.gguf \
  WORK=/tmp/slha-llama \
  CALIB_DIR=/tmp/slha-llama/calibration \
  WEIGHTS_DIR=/tmp/slha-llama/weights \
  integration/llama.cpp/scripts/train_layer_weights.sh mixed

# 7. Round-trip perplexity
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh roundtrip

# 8. Shadow-score quality gate (attention unchanged)
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh shadow

# 9. Strict direct compressed-score replacement (fails closed)
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh replace
```

All scripts pin the llama.cpp tag (`b9860`) and verify the commit hash before
building.

## Runtime interface

Modes are selected via environment variables:

| Variable           | Values                              |
|--------------------|-------------------------------------|
| `SLHA_KV_MODE`     | `off` / `passthrough` / `collect` / `roundtrip` / `tilestore` |
| `SLHA_SCORE_MODE`  | `off` (default) / `shadow` / `replace` |
| `SLHA_CODEC`       | `mixed` (default) / `mix3` / `grouped` / `nf4` / `tq3` |
| `SLHA_WEIGHTS_DIR` | directory with `layer-NNN.slhw` and `manifest.json` |
| `SLHA_CALIBRATION_NONFINITE_POLICY` | `reject` (default) / `drop-row` (research/recovery only) |
| `SLHA_CALIBRATION_MIN_ROWS` | minimum rows a layer must retain under `drop-row` (default 1) |

Shadow and replace modes require `SLHA_KV_MODE=tilestore` so the K tiles are
encoded at the K-cache write seam. Both force `--flash-attn off --parallel 1`
so the baseline logits are materialised and the tile-store positions stay
contiguous within a single sequence. Replace mode additionally pins
`--batch-size 512`.

## Research: layerwise SLHA score-quality diagnosis

This section diagnoses **which layers** the direct compressed-score replacement
path degrades and **which score distortion** predicts that degradation. It is a
diagnostic milestone — the production score mathematics are unchanged.

Provenance: SLHAv2 `23e27c0`, llama.cpp `fdb1db877c526ec90f668eca1b858da5dba85560`,
Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 test, mixed codec, weights from the
corrected PR #59 pipeline. Full data + hashes:
[`results/layerwise_score_gap.json`](results/layerwise_score_gap.json).

### Experimental layer-mask interface

`SLHA_SCORE_LAYERS` selects which layers use direct SLHA score replacement;
unselected layers pass baseline Q·K through unchanged.

| Spec | Meaning | | Spec | Meaning |
| --- | --- | --- | --- | --- |
| `all` | every layer (default) | | `0-6` | inclusive range |
| `none` | no layer (only valid empty) | | `0-3,7,12-14` | combined |
| `7` / `3,7,12` | single / list | | (invalid) | fails closed → `valid=false` |

Parsing is strict: negatives, out-of-range ids (checked against the model layer
count), malformed ranges, and an empty spec are rejected; duplicates are
de-duplicated deterministically. `SLHA_SCORE_MASK_SUMMARY` reports the requested,
resolved, and executed masks plus per-selected-layer coverage.

### Method

Screening ran at **4 chunks** (reduced from 12 — this session's host was
materially slower); the pass-through control used the same 4 chunks, so every
delta is consistent. Deltas are versus the **pass-through custom-op control**
(B). One run per configuration (screening).

### Controls and headline

| Control | mean PPL (4 chunks) | Δ vs B |
| --- | ---: | ---: |
| A unpatched baseline | 9.3852 | — |
| B pass-through custom op | 9.3852 | 0.0000 |
| C all-layer replacement | 13.5421 | **+4.1569 (+44.3%)** |

The custom op is inert (A = B); all-layer replacement reproduces PR #58's ~+42 %
gap at the reduced chunk count.

**The degradation is distributed and super-additive.** The five most-damaging
single layers are 5 (+0.577), 0 (+0.253), 12 (+0.150), 8 (+0.122), 24 (+0.121);
the five least are 21 (+0.020), 27 (+0.007), 18 (+0.006), 25 (−0.057), 3
(−0.065) — two layers *improve* PPL alone. No single layer dominates: the largest
is 0.58 of the 4.16 total. The **sum of the 28 single-layer deltas is 2.245**,
but all-layer replacement is **4.157** — a **1.85× super-additive** amplification:
errors compound through the residual stream.

Cumulative prefixes confirm progressive, mid-network-weighted accumulation:

| Prefix | ΔPPL | | Prefix | ΔPPL |
| --- | ---: | --- | --- | ---: |
| `0` | +0.253 | | `0-13` | +2.853 |
| `0-3` | +0.393 | | `0-20` | +3.773 |
| `0-6` | +1.546 | | `0-27` | +4.157 |

The largest increments fall in the middle of the stack (layers ~4–20).
Cumulative **suffixes** (from the top of the stack) confirm the front-loading:
`14-27` (last 14 layers) is only +0.579 while `0-13` (first 14) is +2.853 — the
**first half causes ~5× the damage of the second half**. The four quartiles
`0-6 / 7-13 / 14-20 / 21-27` contribute **+1.55 / +0.49 / +0.45 / +0.15**: damage
is concentrated in the **first quartile** and tapers toward the output.

### Best predictor of PPL damage (exploratory, 28 layers)

Correlating single-layer PPL damage against per-layer raw-score shadow metrics:

| Metric | Pearson | Spearman |
| --- | ---: | ---: |
| **top-1 attention agreement** | **−0.42** | **−0.51** |
| top-5 overlap | −0.42 | −0.40 |
| MAE | +0.37 | +0.50 |
| cosine | +0.01 | −0.34 |
| relative-L2 | −0.01 | +0.17 |

**Top-1 agreement is the best predictor** (Spearman −0.51): damage tracks how
often SLHA's *argmax* key differs from baseline's, not the overall score
correlation. Raw-score cosine (~0.99 everywhere) does **not** predict damage.
With only 28 layers these are exploratory (moderate |ρ|).

### Score-path semantics (audit)

The op replaces logits after the raw `kq = Q·K` and **before** `soft_max_ext`, so
for both paths the `1/sqrt(head_dim)` scale, causal mask, RoPE positional
information, GQA head mapping, and (absent) softcap are **identical**. Qwen2.5 has
no attention logit softcap; there is no additive positional bias; baseline and
SLHA use identical active KV lengths (padded positions are exactly zero and
masked). **Only two things differ:** the score approximation itself, and that the
*same fixed* `1/sqrt(head_dim)` scale is applied to SLHA scores whose magnitude
is not calibrated to Q·K. If `E[|slha|] ≠ E[|Q·K|]` the effective softmax
temperature differs — the leading mechanism hypothesis for the damage, consistent
with top-1 (argmax) agreement being the best damage predictor.

### Limitations

Single-run screening at a reduced chunk count (not a formal noise floor);
28-layer correlations are exploratory. The finer post-softmax / affine /
per-head / position diagnostics were left to out-of-tree instrumentation
(archived + hashed in the results JSON); that instrumentation segfaulted on this
host and did not emit metrics, so the post-softmax analysis rests on the
committed raw-score shadow metrics (top-1 agreement being the argmax/attention
selection signal) and the score-path audit rather than on measured softmax
divergences. Absolute PPLs are not comparable to the 12-chunk PR #58 numbers, but
within-experiment deltas are. The root cause is not proven: the
temperature-mismatch hypothesis needs an end-to-end scale experiment, and
super-additivity means single-layer screening understates joint damage.
That end-to-end scale experiment was subsequently run and **rejected** the
temperature hypothesis — see
[Research: score-temperature (magnitude) calibration](#research-score-temperature-magnitude-calibration)
below.

## Research: score-temperature (magnitude) calibration

PR #60 left one mechanism hypothesis open: that the direct compressed-score
degradation is primarily a **layer-dependent score-magnitude mismatch**,
equivalent to an incorrect softmax temperature. This section tests it end to end
and **rejects it**. Production score mathematics are unchanged; scaling is a
strict, default-off (`a = 1.0`) experimental knob.

Provenance: llama.cpp `fdb1db877c526ec90f668eca1b858da5dba85560` (tag b9860),
Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 raw test, mixed codec, weights from the
PR #59 pipeline. Full data, per-layer tables and hashes:
[`results/score_temperature_calibration.json`](results/score_temperature_calibration.json).

### The knob, and exactly what it is

The op replaces logits after the raw `kq = Q·Kᵀ` (`llama-graph.cpp:2451`) and
**before** `soft_max_ext` (`:2500`). For Qwen2.5 there is no logit softcap, no
`kq_b`, no ALiBi and no sinks, and `kq_scale = 1/√128`, so

```
baseline:  P = softmax( kq_scale · (Q·Kᵀ) + mask )
replace:   P̂ = softmax( kq_scale ·  S_SLHA + mask )
scaled:    P̃ = softmax( (kq_scale · a_layer) · S_SLHA + mask )
```

`a_layer` multiplies the **effective inverse temperature**, applied exactly once
to the raw score; the `1/√head_dim` factor is applied afterwards by
`soft_max_ext` and is never applied twice. Because softmax is shift-invariant per
row, `a·s = a·(s − s̄_row) + (a−1)·s̄_row` and the row-constant term is discarded —
so scaling the raw score *is* scaling the row-centred logit, and a global sweep
probes the softmax-relevant temperature directly.

### Determinism had to be fixed first

The replacement path contained a data race: `dst` was zero-initialised by flat
element range but written by vector range, and a ggml custom op has no internal
barrier, so one worker could blank rows another had already filled. Three
**numerically identical** replacement configurations returned
13.4196 / 13.4068 /
13.4237 — a spread of 0.0169 PPL.
After the fix they return bit-identical values. Every measurement below was taken
with the fixed binary, so differences between configurations are signal, not noise.

| identity-equivalent configuration | PPL (4 chunks) |
| --- | ---: |
| id_replace_noscale | 13.4162 |
| id_replace_g1p0 | 13.4162 |
| id_replace_file1p0 | 13.4162 |

The pass-through custom-op control and the SLHA-off baseline both give
9.3852 — the op itself is inert. Unscaled replacement gives
13.4162, a gap of **4.0310 (42.95%)**.

### Offline fit: the magnitudes are already calibrated

Shadow mode streams sufficient statistics over causally-unmasked positions
(`k ≤ t`, clamped to written tiles). The pair count per layer matches the analytic
prediction Σ(t+1)·heads·chunks **exactly**, proving no vector was skipped and that
`t` is the true token position.

| estimator | min | median | max |
| --- | ---: | ---: | ---: |
| OLS through origin | 0.9828 | 0.9968 | 1.0036 |
| robust median-ratio | 0.9886 | 0.9886 | 1.0116 |
| variance matching | 0.9999 | 1.0097 | 1.0260 |
| slope with free intercept | 0.9963 | 1.0002 | 1.0079 |
| Pearson r(b,s) | 0.9795 | 0.9908 | 1.0000 |

Fitting with a free intercept — the softmax-relevant form, since a constant
per-row offset cancels — puts every layer within
**0.79%** of identity, tighter than the
through-origin fit. So the near-unit scale is not an artifact of forcing the fit
through the origin. Applying the best-fit scale removes only
**0.38%** of the score's squared error
(pair-weighted; 1.00% equally
weighted across layers, at most 5.88% for
any single layer).

The fit is not dominated by a degenerate subpopulation: each of the 12 heads
carries exactly 1/12 of the samples with per-head scales inside a ~3% band,
position-bucket counts follow the causal prediction with per-bucket scales ≈ 1.0,
near-zero scores carry a vanishing share of the OLS denominator, and there are no
non-finite pairs. Two caveats are recorded rather than glossed: a **pooled**
cross-layer OLS is meaningless here because layer 0 alone carries
99.91% of Σs², and the robust estimator is quantised to
its 0.01-dex histogram bin (≈2.3%), so its digits beyond ~1% are not meaningful.

**Per-layer structure is not resolved.** The split-half disagreement (max 0.0256) is AS LARGE AS or LARGER than the entire per-layer spread of the fitted scale (0.0208), so the apparent per-layer structure is NOT resolved: the per-layer scales are consistent with all layers sharing a scale of ~1.0 plus estimation noise. This makes the fitted per-layer scale files perturbations of the identity rather than a genuine search over per-layer temperatures, and it is why they are reported alongside, not instead of, the direct global sweep. Rank correlation between the two disjoint halves is 0.354.

### End-to-end: no temperature recovers the gap

Every measured point, at 4 chunks. `recovered_gap = (unscaled − config) / (unscaled − pass-through)`.

| global scale a | PPL | recovered gap |
| ---: | ---: | ---: |
| 0.40 | 269.9862 | -6364.92% |
| 0.50 | 69.4502 | -1390.08% |
| 0.60 | 26.8259 | -332.66% |
| 0.70 | 17.6512 | -105.06% |
| 0.80 | 14.6611 | -30.88% |
| 0.90 | 13.8585 | -10.97% |
| 0.92 | 13.7724 | -8.84% |
| 0.94 | 13.7056 | -7.18% |
| 0.96 | 13.5451 | -3.20% |
| 0.98 | 13.4530 | -0.91% |
| 0.99 | 13.5311 | -2.85% |
| 1.00 | 13.4162 | 0.00% |
| 1.01 | 13.6068 | -4.73% |
| 1.02 | 13.4988 | -2.05% |
| 1.04 | 13.5626 | -3.63% |
| 1.06 | 13.4889 | -1.80% |
| 1.08 | 13.4590 | -1.06% |
| 1.10 | 13.5629 | -3.64% |
| 1.20 | 13.8681 | -11.21% |
| 1.30 | 14.0534 | -15.81% |
| 1.50 | 15.2022 | -44.31% |
| 1.75 | 17.2626 | -95.42% |
| 2.00 | 19.8388 | -159.33% |

A clean U-curve with its minimum at **a ≈ 1.0**. Both directions are worse:
sharpening degrades steadily, flattening degrades catastrophically (a = 0.40 →
269.99),
which is what rank-preservation predicts: flattening destroys the selectivity the
model depends on while buying nothing back.

**Resolution floor.** The binary is deterministic, yet PPL(a) is not smooth at
fine scale: a least-squares quadratic through the local window
[0.90, 1.10] leaves a residual roughness of
**0.0584 PPL RMS** (max 0.1143). That is not
measurement noise — identical configurations give bit-identical results — but genuine
chaotic sensitivity of the forward pass to tiny attention perturbations. It sets the
resolution for any recovered-gap claim at about
**1.45%** of the gap. The smooth
component of the curve has its minimum at a = 1.037 with curvature
d²PPL/da² ≈ 40.4; no measured point recovers any of the gap.

| fitted-scale strategy | PPL | recovered gap | manifest |
| --- | ---: | ---: | --- |
| `sc_robust_0_6` | 13.4196 | -0.08% | 26239a1bcb0c |
| `sc_robust_all` | 13.4434 | -0.67% | 12799dc02ca2 |
| `sc_global_robust` | 13.4766 | -1.50% | 5858defc1dfa |
| `sc_ols_0_13` | 13.4821 | -1.63% | 34c2920dc7f3 |
| `sc_ols_all` | 13.5006 | -2.09% | 820c9a847428 |
| `sc_ols_0_20` | 13.5125 | -2.39% | f910a38c8a09 |
| `sc_ols_0_6` | 13.5192 | -2.56% | 6a95ebf3689e |
| `sc_var_all` | 13.5216 | -2.61% | ae9af62c6611 |
| `sc_global_ols` | 13.5497 | -3.31% | b4c6fa1ba027 |

### Twelve-chunk validation (3 repetitions each)

| configuration | mean PPL | sample stdev | spread | recovered gap | reps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v_passthrough` | 11.8644 | 0.0000 | 0.0000 | — | 3 |
| `v_replace_noscale` | 16.8855 | 0.0000 | 0.0000 | 0.00% | 3 |
| `v_replace_g1p0` | 16.8855 | 0.0000 | 0.0000 | 0.00% | 3 |
| `v_best_global` | 16.8855 | 0.0000 | 0.0000 | 0.00% | 3 |
| `v_best_perlayer` | 16.9073 | 0.0000 | 0.0000 | -0.43% | 3 |
| `v_best_early` | 16.9138 | 0.0000 | 0.0000 | -0.56% | 3 |

### Why this excludes the whole family, not just the points tested

Softmax normalises over one `(layer, head, query)` row, so a per-layer, per-head,
per-query-row or context-length-dependent positive scale is **constant within the
normalisation group**. Every such scale is exactly rank-preserving, and the global
sweep measures the family's mean directly. The residual bounds its spread:
attributing **100%** of each layer's score residual to per-row gain jitter — the
most generous possible magnitude hypothesis — gives σ_a ≈ 0.070, which
priced against the measured sweep curvature
(d²PPL/da² ≈ 40.4) costs only
≈ 0.100 PPL, about
2.5% of the gap.
Explaining the whole gap would need a jitter several times larger than the one
actually measured. The scale family is excluded numerically, not merely unsampled.

### Conclusion

Per-layer and global multiplicative calibration recovered less than 1% of the PPL gap, while fitted scales remained close to identity and scaling removed less than approximately 1% of raw-score error. No positive rescaling that is constant within a softmax row -- per-layer, per-head, per-query-row, or context-length-dependent -- can close the gap: every such rescaling is exactly rank-preserving, the measured optimum of the global sweep sits at a ~ 1.0, and even attributing 100% of the score residual to per-row gain jitter prices that whole family at only a few percent of the gap. The quality gap is therefore dominated by score distortion that is NOT constant within a softmax row. This experiment does not further decompose that residual into reordering versus order-preserving gap error.

Twelve-chunk validation repeats were **bit-identical** (sample stdev 0.0000 across
three repetitions of every configuration), and determinism was confirmed on the
genuinely scaled write path as well, not only on the `a = 1.0` path that skips the
scaling loop.

What this does **not** establish: it does not decompose the residual into
reordering versus order-preserving gap error. A monotone nonlinear magnitude map
is non-scalar, order-preserving, changes the softmax distribution, and is
invisible to every statistic computed here; a per-**key** magnitude error is
likewise not row-constant and therefore does change ranking. The `top1`/`top5`
figures quoted in PR #60 and reproduced by the shadow metrics are computed over
the full `n_kv` row including causally-masked keys, so they are not clean
attention-relevant rank-agreement numbers and are not used as load-bearing
evidence here.

### Recommended next experiments

After a negative scaling result the next diagnostic should target the score
*ordering* and per-key magnitude, not another temperature sweep:
- joint Q/K projection training (optimize the compressed projection for score ordering)
- pairwise ranking loss on score pairs rather than L2 on score values
- top-k preservation loss during projection training
- per-head projection calibration
- residual correction of SLHA scores (learned low-rank correction term)
- hybrid exact top-k plus compressed tail scoring

None of these are implemented in this PR.

### Experimental interface

`SLHA_SCORE_SCALE` (`"0.75"` or `"layer:0=0.91,5=0.72"`) and
`SLHA_SCORE_SCALE_FILE` (JSON `{"global":…,"layers":{…}}`) set the per-layer
scale; default `1.0` (no-op). Strict and fail-closed: only finite strictly
positive scales; zero, negative, NaN, ±Inf, malformed, duplicate/out-of-range
layer ids, and — in per-layer mode — any selected layer lacking a scale are
rejected and mark the run invalid. `SLHA_SCORE_SCALE_SUMMARY` reports the
requested and resolved scales, a manifest SHA-256, `scaled_vectors`/
`scaled_logits`, `invalid_scale` and `scale_manifest_valid`.
`SLHA_SCALE_FIT_JSON=<path>` (shadow mode) writes the offline per-layer fit.

### Limitations

Screening ran at 4 chunks because this host is materially slower than the PR #58
host; every control used the same chunk count, so within-experiment deltas are
consistent, but absolute PPLs are not comparable to the 12-chunk PR #58 numbers.
The offline fit is off-policy — it estimates scales on baseline-conditioned
states with no cross-layer error compounding — so only the end-to-end sweep is
on-policy; that is why the sweep, not the fit, carries the conclusion. The
per-layer scale files tested are magnitude-fitted perturbations of the identity
(all within ~3%), not a PPL-optimised search of the 28-dimensional scale space.

## Research: rank-transplant oracle (ranking vs order-preserving error)

PR #62 established that no rescaling constant within a softmax row explains the
compressed-score quality gap. That left two mechanisms unseparated: **(A) key-ranking
errors** — SLHA puts the wrong keys at the top — and **(B) order-preserving errors** in
the score gaps and shape. This section separates them with a diagnostic transplant
oracle. It is measurement only: production score mathematics are unchanged and the
oracle is default-off.

Provenance: source content proven byte-identical to squash merge `e8db022`,
llama-perplexity `43dc88a072b590d6…`, libllama `ea4979aa14ff6c68…`,
model `d7efb072e7724d25…`, dataset `136677b69515d194…`,
weight manifest `afe6deb0cf986015…`. Full data, per-layer tables and
hashes: [`results/rank_transplant_oracle.json`](results/rank_transplant_oracle.json).

> **These oracles require access to exact baseline scores and are not deployable
> replacement methods.** Every one of them reads the baseline `Q·Kᵀ` row that a compressed
> KV cache exists precisely to avoid computing. They bound what a better codec could
> achieve; they are not one. The oracle is default-off and no production path changes.

### The oracles

At the raw `kq = Q·Kᵀ` seam, for query token `t`, the oracle rewrites the row from the
paired baseline row `B` and SLHA row `S`:

```
Oracle A   baseline ranking  +  the visible SLHA value multiset
Oracle B   SLHA ranking      +  the visible baseline value multiset
top-k      baseline's top-k keys promoted; the tail keeps its relative SLHA order
```

**Causal visibility is part of the definition.** Query `t` attends keys `0..t`, so the
oracle domain is `n_visible = min(n_check, t+1)`. An earlier version spanned the whole
written prefix; masked keys then consumed value-multiset slots that softmax never sees,
which diluted the transplant. At the four-chunk screening resolution, fixing it moved
Oracle A from 10.4088 to
10.9429 PPL and the recovered fraction from
74.6% to
61.4% — a material change, not a rounding
difference. Every pre-fix oracle run was quarantined, not reinterpreted.

Values are gathered through the *other* vector's own ranking permutation, so the
transplanted multiset is bit-exact. The ordering invariant is **weak-order** preserving:
the output must be non-increasing along the reference permutation. A strict
permutation-equality check is wrong whenever the transplanted values tie —
`B = [1.0, 9.0]`,
`S = [5.0, 5.0]` gives
`out = [5.0, 5.0]`, whose rank vector
`[0, 1]` cannot equal the reference
permutation `[1, 0]` because the two values are equal.

### Identity controls

Nothing below is interpreted unless both identity controls are exact at twelve chunks.

| control | PPL (12 chunks, mean of 3) | must equal |
| --- | ---: | --- |
| pass-through | 11.8644 | — |
| baseline-identity oracle | 11.8644 | pass-through |
| strict replacement | 16.8855 | — |
| SLHA-identity oracle | 16.8855 | strict replacement |

Both hold exactly (`controls_exact = true`). Three repetitions of each of the
nine configurations are **bit-identical**, sample standard deviation exactly zero, with
identical strict, oracle, exact-tie, invariant-check and active-domain accounting
counters (`all_deterministic = true`). Any difference would be a regression after the
PR #62 race fix and would block aggregation.

### Result

Total gap at twelve chunks: 16.8855 − 11.8644 = **5.0211 PPL**.

| oracle | PPL | change vs strict | fraction of gap |
| --- | ---: | ---: | ---: |
| Oracle A — baseline ranking, visible SLHA values | 13.5756 | +3.3099 | 65.92% |
| Oracle B — SLHA ranking, visible baseline values | 19.6719 | -2.7864 | -55.49% |
| order-preserving residual (Oracle A − pass-through) | — | +1.7112 | 34.08% |

Restoring the baseline ordering of causally visible keys while preserving the causally
visible SLHA score-value multiset recovers 65.9% of the gap. The remaining
1.7112 PPL is the **order-preserving residual** — error that survives with
the ranking already correct. It is *not* labelled magnitude error: PR #62 rejected the
magnitude hypothesis, and this experiment does not identify what the residual is.

Oracle B is reported independently and read conservatively: applying the baseline value
distribution to the incorrect SLHA key ordering **amplifies** the quality loss. This
demonstrates a strong interaction between key assignment and score geometry; it does not
prove that score geometry is irrelevant. **Oracle A and Oracle B are not an additive
decomposition** — their fractions do not sum to 1 and must not be read as a variance
partition.

### How many keys have to be right

| restoration | PPL | absolute recovery | fraction of gap | fraction of full Oracle A | remaining residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| top-1 | 14.2400 | +2.6455 | 52.69% | 79.93% | 2.3756 |
| top-16 | 13.6278 | +3.2577 | 64.88% | 98.42% | 1.7634 |
| full ranking (Oracle A) | 13.5756 | +3.3099 | 65.92% | 100.00% | 1.7112 |

- CONFIRMED: top-1 restoration alone reaches 79.93% of the full Oracle A recovery.
- CONFIRMED as approximate saturation: top-16 reaches 98.42% of the full Oracle A recovery.

The SIGN of the top-16 minus full-ranking difference reversed between the two resolutions. At four chunks top-16 was 0.0113 PPL BELOW full ranking; at twelve chunks it is 0.0522 PPL ABOVE it. The screening appearance that top-16 slightly exceeded full ranking therefore does NOT persist: at the validated resolution top-16 recovers slightly less than the full ranking, as expected. Both differences are small in absolute terms, but the reversal is why the four-chunk contrast was never promoted to a claim.

### Early layers

| restoration | PPL | recovery | fraction of gap |
| --- | ---: | ---: | ---: |
| full-rank Oracle A, layers 0–6 | 12.4770 | +4.4085 | 87.80% |
| full-rank Oracle A, all 28 layers | 13.5756 | +3.3099 | 65.92% |

Correcting rankings only in early layers can outperform correcting every layer because some later-layer SLHA deviations may be compensating, while early ranking errors trigger downstream amplification. This does not mean later layers are irrelevant.

### Screening versus final

The twelve-chunk matrix is authoritative. The four-chunk sweep is screening evidence only.

| metric (recovered fraction of gap) | four-chunk screening | twelve-chunk final | absolute difference | direction preserved |
| --- | ---: | ---: | ---: | :---: |
| Oracle A recovery | 61.36% | 65.92% | +4.56 pp | yes |
| Oracle B change | -37.26% | -55.49% | -18.23 pp | yes |
| top-1 recovery | 46.98% | 52.69% | +5.71 pp | yes |
| top-16 recovery | 61.64% | 64.88% | +3.24 pp | yes |
| early 0-6 recovery | 88.55% | 87.80% | -0.75 pp | yes |

Every direction is preserved (`all_directions_preserved = true`). The
magnitudes move by a few percentage points, which is why the four-chunk figures were never
promoted to validated claims.

### Layerwise screening, including the anomalies

The 28 matched pairs (`U_layer_L` strict vs `L_oracleA_L` oracle, identical conditions)
are a **four-chunk screening** result; no equivalent twelve-chunk layerwise decomposition
was run. Classification precedence, applied in this order:

1. `unstable_denominator` — |damage| < eps -- checked first, because every fraction below is undefined when the denominator is not resolvable
1. `negative_damage` — damage <= -eps -- replacing this layer alone IMPROVES PPL, so a 'recovered fraction' has no meaning regardless of the oracle change
1. `over_recovery` — damage >= eps and oracle_change > damage (fraction > 1) -- the oracle drives PPL below pass-through; the fraction is reported uncapped
1. `negative_recovery` — damage >= eps and oracle_change < 0 -- the transplant makes this layer worse; the negative fraction is reported signed, never clipped
1. `positive_recovery` — everything else

With a declared denominator guard of ε = 0.01 PPL:

- **negative damage** (replacing the layer alone *improves* PPL, so a fraction is meaningless): layers 3, 25
- **unstable denominator** (|damage| < ε): layer 10
- **positive damage but negative recovery** (the transplant makes the layer worse; the signed fraction is reported, never clipped): layers 1, 2, 8, 9, 15, 18, 22
- **over-recovery** (fraction > 1): none — no screening layer recovers more than 100%

No fraction is capped anywhere in the artifact. Layer 5 carries the largest single-layer
damage (0.5578 PPL) and the largest absolute recovery
(+0.4220 PPL), but its recovery does not exceed its damage: the
fraction is 0.7565 and the oracle result 9.5210 stays above
pass-through 9.3852. The largest fraction observed is
0.9907 at layer 27.

Summing the **signed** single-layer absolute recoveries over layers 0–6 — including the
negative terms, none excluded — gives 0.3811 PPL, while restoring the group
jointly recovers 3.5693 PPL: an interaction of
**+3.1882 PPL**. Early ranking errors amplify downstream. This is a
screening figure and is labelled as such in the artifact.

### Active-key rank agreement

All statistics are restricted to **causally visible** keys — written tile, in-stream,
finite, unmasked. They supersede the PR #60 top-1/top-5 figures, which were computed over
the full padded row and are not attention-relevant. Micro (pooled over rows) and macro
(equal weight per layer) are reported separately and never averaged together.

| statistic | micro | macro mean | macro median | macro min | macro max |
| --- | ---: | ---: | ---: | ---: | ---: |
| top-1 agreement | 0.7389 | 0.7389 | 0.8159 | 0.2093 | 1.0000 |
| top-2 overlap | 0.7548 | 0.7548 | 0.8004 | 0.4597 | 0.9173 |
| top-4 overlap | 0.8055 | 0.8055 | 0.8054 | 0.6361 | 0.9435 |
| top-8 overlap | 0.8384 | 0.8384 | 0.8465 | 0.7182 | 0.9506 |
| top-16 overlap | 0.8627 | 0.8627 | 0.8703 | 0.7704 | 0.9619 |
| Spearman (fractional ranks) | 0.9640 | 0.9640 | 0.9644 | 0.9267 | 0.9934 |
| Kendall tau-b | 0.8529 | 0.8529 | 0.8484 | 0.7807 | 0.9383 |

**Tie counts are sampled diagnostics, not exhaustive.** The metrics hook records a row only
when `(t % 16) == 0 && h < 2`, so 7,224 rows were inspected
(258 per layer across 28 layers) — 1.049% of the
688,800 replaced vectors in a full-model run. On that sample:
96 baseline exact-tie pairs, 70 SLHA exact-tie pairs,
280 undefined rows and top-k boundary ties
top1=0, top2=0, top4=0, top8=0, top16=1 — a boundary tie means the
k-th and (k+1)-th baseline scores are exactly equal, so top-k membership is decided by the
deterministic index tiebreak rather than by the score. These are **not** comparable with the exhaustive oracle
counter `oracle_ties = 12511`, which counts tied comparisons over every replaced vector
during permutation construction — a different population and a different definition. Do
not divide one by the other.

Position accounting holds on every sampled row (`physical = included + causally_masked +
padding + inactive_stream + nonfinite`), with 0 failures.

### What this does and does not establish

Restoring the baseline ordering of causally visible keys while preserving the causally visible SLHA score-value multiset recovered most of the measured quality gap. Correcting only a small number of highest-ranked keys captured a large fraction of that benefit, and correcting layers 0-6 prevented substantial downstream amplification.

The dominant measured mechanism is therefore incorrect assignment of high score values to visible keys. A substantial order-preserving residual remains, and Oracle B demonstrates that key assignment and score geometry interact strongly.

It does **not** establish that:

- ranking explains all degradation
- the residual is purely magnitude error
- later layers do not matter
- Oracle A and Oracle B partition the gap additively

These oracles require access to exact baseline scores and are not deployable replacement methods. Every oracle in this experiment reads the baseline Q·K row that a compressed KV cache exists precisely to avoid computing; the measurements bound what a better codec could achieve, they do not constitute one.

### Limitations

- one model (Qwen2.5-1.5B-Instruct Q8_0), one corpus (WikiText-2 raw test), one codec configuration; nothing here is shown to generalise
- twelve chunks of 512 tokens is 6144 tokens -- enough to separate these effects, far short of a full benchmark
- the layerwise decomposition and the early-layer interaction are FOUR-CHUNK screening results; no twelve-chunk layerwise matrix was run
- the active-key statistics come from a deterministic 1-in-16 token, 2-of-12 head sample, not an exhaustive pass
- perplexity is the only quality metric; no downstream task was measured
- the order-preserving residual is quantified but not explained -- this experiment does not identify what it consists of
- two process-tree terminations remain unexplained; they were reconciled by re-running, not by root-cause analysis

### Recommended next experiment

Constrain the codec so that the top-k baseline ordering is preserved by construction and measure how much of the 65.92% ranking recovery survives without any baseline-score access. The top-1 result (79.93% of the full ranking benefit from a single key) and the layers 0-6 result (87.80% of the gap from seven layers) together suggest the cheapest viable target: an early-layer, top-k-preserving score path. A parallel line should attack the order-preserving residual directly, since PR #62 already ruled out a per-row rescaling constant and this experiment shows the residual is 34.08% of the gap even with the ranking exactly correct.

### Methodological corrections

Every defect found during this experiment is recorded in the artifact under
`provenance.experimental_provenance_chronological`. The two that changed a measurement or
a decision:

1. **Incorrect strict-permutation invariant.** The ordering check required exact
   permutation equality, which is only valid when no two transplanted values tie. Corrected
   to a weak-order check. Observational: the quarantined runs reproduced identical PPL.
2. **Non-causal oracle span.** The oracle spanned the whole written prefix instead of the
   causally visible one. Material: it inflated the apparent ranking contribution. All
   affected runs were quarantined and re-run, never reinterpreted.

Also recorded there: the four pre-lineage twelve-chunk attempts that were quarantined and
re-run once the harness recorded the full lineage block; the `running`-state PID recording
fix; and the two unexplained process-tree terminations.

### Resolution

No single scalar "experimental resolution" is claimed. The relevant quantities have
different sources and are reported separately:

- deterministic PPL printing precision: 0.0001 (llama-perplexity prints the final estimate to four decimal places)
- twelve-chunk repetition spread: 0.0000 PPL maximum across all nine configurations (all bit-identical)
- screening chunk-count limitation: 4 chunks × 512 tokens = 2048 tokens evaluated
- smallest screening contrast treated as meaningful: 0.0108 PPL (layer 27 single-layer damage, the smallest damage still classified stable)
- The top-16 and full-ranking screening values differ by 0.0113 PPL, which is small relative to the coarse four-chunk screening protocol. The twelve-chunk validation is used to determine whether this difference persists. It did not persist with the same sign: at twelve chunks top-16 sits 0.0522 PPL ABOVE full ranking rather than below it. Top-16 still reaches 98.42% of the full Oracle A recovery, so approximate saturation holds, but the four-chunk contrast was too fine for its sign to be meaningful.

### Execution integrity

Two completeness gates run independently; neither implies the other.

- screening: 75/75 accepted and revalidated, two agreeing reads under proven quiescence, report `526f6f5da91352f5…`
- twelve-chunk: 27/27 accepted and revalidated, two agreeing reads, report `5979a43b5253f723…`, bound by sha256 to the frozen screening report

Acceptance is validated from the run itself, never inferred from a results row. Each
attempt has an immutable log and parsed summary, an explicit child wait, exit and signal
status, and an atomically written state record. Twelve-chunk states additionally require
a complete lineage block and the exact declared run geometry. Resume revalidates every
accepted state before skipping it. Superseded and interrupted attempts are preserved as
provenance and never counted toward completeness.

On 2 occasions the process tree was unexpectedly terminated. Available evidence
did not identify an OOM, disk-capacity failure or application crash
(15,209 MB of 16,075 MB free; disk 35% used; no OOM record; no crash trace), so the cause is recorded as **unknown**. Both
were reconciled by preserving the interrupted attempt as provenance and launching a new
immutable attempt.

## Earlier round-trip results

Recorded under the earlier protocol; see git history and
[`results/measurements.json`](results/measurements.json) for details.

| Mode        |      PPL | ΔPPL absolute | ΔPPL relative | Notes           |
| ----------- | -------: | ------------: | ------------: | --------------- |
| baseline    | 11.8753  |             — |             — | original        |
| passthrough | 11.8753  |          0.00 |          0.0% | hook sanity     |
| mixed       | 16.5976  |          4.72 |         39.8% | SLHA round-trip |
| mix3        | 16.6460  |          4.77 |         40.2% | SLHA round-trip |

The round-trip callback reconstructs K from the SLHA tile using the latent plus
a linear estimate of the sign-LSH residual. The 256-bit residual sketch is a
*score-side* correction rather than a faithful inverse of the quantization
error, and the round-trip perplexity reflects that limitation.

## Limitations and known issues

* This milestone evaluates **score quality only**. No physical K-cache memory
  reduction is claimed and no fused attention kernel is implemented.
* Baseline Q·K is still materialised by llama.cpp; the custom operation replaces
  logits after the baseline matrix multiplication.
* Strict replace mode supports only one parallel sequence (`n_stream == 1`).
* Calibration collection can intermittently emit non-finite K rows (see
  **Calibration integrity policy**). This is now caught: the collection driver
  and the trainer both fail closed on any non-finite row under the default
  `reject` policy, so poisoned dumps can no longer reach training silently. The
  precise upstream trigger inside llama.cpp's priming decode is characterised
  (whole-row, host-dependent) but not instrumentally proven, so the validation
  boundary — not a source-side exclusion predicate — is the fix.
* Numerical PPL results show run-to-run variation even in unpatched llama.cpp
  (observed spread 0.0100 PPL over three runs). Production counters were fully
  deterministic; the source of the numerical variation was not instrumentally
  isolated.
* Statistical power is limited: three runs per group, approximate exploratory
  confidence intervals.
* The measured quality gap is specific to this configuration and its precise
  cause has not been isolated.
* Per-layer projections are trained on K only (`fit_with`), not joint K/Q,
  because the collection seam currently only exposes K.
* AddressSanitizer builds successfully but the runtime shadow memory could not
  be allocated in the current container (`ulimit -v` insufficient). No memory
  errors were observed in normal runs.
* Only CPU inference is exercised here; GPU is out of scope.

## Gates

The Cargo workspace gates still pass:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo build --locked --workspace --all-targets --all-features
cargo test --locked --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps
cargo +1.89.0 check --locked --workspace --all-targets --all-features
```

The production-linked tests (strict-replacement counters, calibration validator,
trainer atomicity, experimental score-scale parsing, and the offline
score-scale fit mathematics):

```bash
make -C integration/llama.cpp/tests clean
make -C integration/llama.cpp/tests test
make -C integration/llama.cpp/tests clean
```

The llama.cpp integration gate:

```bash
WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh baseline
WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh passthrough
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh roundtrip
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh replace
```
