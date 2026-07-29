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
and trainer atomicity):

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
