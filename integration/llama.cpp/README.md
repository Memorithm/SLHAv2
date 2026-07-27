# SLHA v2 × llama.cpp — K-cache quality round-trip

This directory implements and measures the first real-LLM quality-path
integration for SLHA v2: every K vector is encoded to a 128-byte SLHA tile and
decoded back to the original K dimension *before* it is stored in llama.cpp's
normal KV cache. Attention is untouched. The goal is to isolate and quantify
the perplexity cost of SLHA-compressing K on a real model.

> This is **not** the final fused-score bandwidth phase. It does not claim any
> KV memory reduction or attention-speed gain.

## Status

Implemented and measured:

* Inert passthrough hook (proves the custom GGML op does not move perplexity).
* Per-layer K activation collection mode (writes `layer-N-k.bin`).
* Automated per-layer projection training script (outputs `layer-NNN.slhw` +
  `manifest.json`).
* SLHA K round-trip callback using `slha_encode_key` + `slha_decode_key`.
* **Experimental shadow-score quality gate**: `SLHA_SCORE_MODE=shadow` compares
  baseline Q·K logits with direct SLHA scores while leaving attention output
  unchanged.
* Reproducible build/apply/measure scripts.

Measured conclusion on the chosen configuration: **the round-trip path does not
preserve perplexity within the pre-registered ≤1 % gate** (ΔPPL ≈ +40 %). See
[`results/measurements.json`](results/measurements.json).

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
├── shim/slha_llama.cpp               # C++ shim (collect / passthrough / roundtrip / tilestore / shadow)
├── shim/slha_llama.hpp
├── scripts/prepare_calibration.sh    # build a separate calibration corpus
├── scripts/train_layer_weights.sh    # train one .slhw per layer
└── results/
    ├── measurements.json             # machine-readable results
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

# 6. Train per-layer projections (mixed codec)
WORK=/tmp/slha-llama \
  CALIB_DIR=/tmp/slha-llama/calibration \
  WEIGHTS_DIR=/tmp/slha-llama/weights \
  integration/llama.cpp/scripts/train_layer_weights.sh mixed

# 7. Round-trip perplexity
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh roundtrip
SLHA_CODEC=mix3 WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh roundtrip

# 8. Shadow-score quality gate (attention is unchanged; prints SLHA-vs-baseline metrics)
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh shadow
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

Shadow mode requires `SLHA_KV_MODE=tilestore` so the K tiles are encoded at the
K-cache write seam; it forces `--flash-attn off --parallel 1` so the baseline
logits are materialised for comparison.

## Measured results

Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 test, 12 chunks, 512 context, 4 threads.

| Mode        |      PPL | ΔPPL absolute | ΔPPL relative | tok/s | Notes           |
| ----------- | -------: | ------------: | ------------: | ----: | --------------- |
| baseline    | 11.8753  |             — |             — | 44.66 | original        |
| passthrough | 11.8753  |          0.00 |          0.0% | 44.81 | hook sanity     |
| mixed       | 16.5976  |          4.72 |         39.8% | 42.28 | SLHA round-trip |
| mix3        | 16.6460  |          4.77 |         40.2% | 42.22 | SLHA round-trip |
| shadow      | 11.8699  |         -0.01 |         -0.0% |  ~48  | score-only gate |

See [`results/measurements.json`](results/measurements.json) for exact SHAs,
commands, and timestamps.

## Why the regression is large

The callback reconstructs K from the SLHA tile using the latent plus a linear
estimate of the sign-LSH residual. While the latent captures the principal
subspace of the calibration K activations, the 256-bit residual sketch is a
*score-side* correction (it preserves attention dot products well in offline
score tests) rather than a faithful inverse of the quantization error. Restoring
the exact per-vector K value from sign bits alone is not what the residual was
designed for, and the perplexity measurement reflects that limitation.

This does **not** rule out the later fused-score path, where attention scores
are computed directly on the compressed tiles without ever reconstructing K.
That path is out of scope for this milestone.

## Limitations and known issues

* Per-layer projections are trained on K only (`fit_with`), not joint K/Q,
  because the collection seam currently only exposes K.
* The round-trip reconstruction uses a spectral residual estimate; it improves
  over latent-only reconstruction but is not an exact inverse.
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

The llama.cpp integration gate:

```bash
WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh baseline
WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh passthrough
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh roundtrip
```
