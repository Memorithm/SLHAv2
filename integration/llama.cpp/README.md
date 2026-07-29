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
exact origin was not instrumentally proven.

The weights were generated after the squash merge from implementation commit
`6361dfdbcd30660bf2d623fe19029938dd209cd7` using an out-of-tree deterministic
non-finite-row removal step.

Per-layer raw and cleaned hashes, aggregate calibration hashes, all weight-file
hashes, the manifest hash and the exact training command are recorded in
[`results/measurements.json`](results/measurements.json).

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
├── tests/replace_strict_tests.cpp    # production-linked strict-counter tests
├── scripts/prepare_calibration.sh    # build a separate calibration corpus
├── scripts/train_layer_weights.sh    # train one .slhw per layer
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

# 6. Train per-layer projections (mixed codec).
#    NOTE: see "Calibration preprocessing" — the collected dumps currently
#    contain non-finite rows that this step does not filter.
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

Shadow and replace modes require `SLHA_KV_MODE=tilestore` so the K tiles are
encoded at the K-cache write seam. Both force `--flash-attn off --parallel 1`
so the baseline logits are materialised and the tile-store positions stay
contiguous within a single sequence. Replace mode additionally pins
`--batch-size 512`.

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
* **The committed collector emits non-finite calibration rows in this
  experiment, and the committed trainer does not reject or filter them. A
  production fix is still required.**
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

The production-linked strict-counter tests:

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
