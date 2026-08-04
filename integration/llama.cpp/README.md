# SLHA v2 × llama.cpp — compressed-score (fused-QK) integration

This directory implements and measures the real-LLM quality path for SLHA v2:
the **fused-QK** mode encodes every K vector to a 128-byte SLHA tile and
**replaces the QK^T attention scores** with SLHA scores computed directly on
the tiles (no K reconstruction). It also keeps the earlier round-trip path
(encode → decode K before storage) as a measured reference.

> This is **not** the final fused-score bandwidth phase: the fused node is a
> CPU GGML callback, so it does not yet claim any KV memory reduction or
> attention-speed gain. This milestone measures *quality* (perplexity).

## Status

Implemented and measured:

* Inert passthrough hook (proves the custom GGML op does not move perplexity).
* Per-layer K activation collection mode (writes `layer-N-k.bin`).
* Automated per-layer projection training script (outputs `layer-NNN.slhw` +
  `manifest.json`).
* SLHA K round-trip callback using `slha_encode_key` + `slha_decode_key`.
* **Fused-QK mode** (`SLHA_KV_MODE=fused`): every K vector is encoded to a
  128-byte tile at `cpy_k` time, and the QK^T attention scores are **replaced**
  by SLHA scores computed directly on the tiles (`slha_prepare_query` +
  `slha_process_tile`). The softmax then operates on the SLHA scores; V is
  untouched. This is the "score compressed tiles directly" path the round-trip
  NO-GO explicitly left open.

Measured conclusions:

* **Round-trip (K reconstruction): NO-GO** — ΔPPL ≈ +40 % (see
  [`results/measurements.json`](results/measurements.json)).
* **Fused-QK: NO-GO** — the SLHA scores do not preserve the attention
  distribution well enough to replace QK^T. Measured on the same config
  (Qwen2.5-1.5B Q8_0, WikiText-2 test, 12 chunks, 512 ctx, 4 threads):
  PPL = **19830** vs baseline **11.88**, with score-diagnostic cos ≈ 0.68 and
  KL ≈ 5.2 on deep layers. The fused path is *functional* (it compiles, runs,
  and demonstrably replaces the scores — see the diagnostics), but the
  quality gate fails massively.

The fused-QK result points to the open problem: the per-layer projections are
trained on K alone (`fit_with`), and the resulting SLHA scores are correlated
with the true QK^T (cos ≈ 0.68) but with a very different softmax
distribution (KL ≈ 5). Closing the gap would require joint K/Q calibration,
score-scale calibration, or a much larger latent/residual budget — all future
work, not validated here.

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
├── shim/slha_llama.cpp               # C++ shim (collect / passthrough / roundtrip)
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

# 8. Fused-QK perplexity (scores computed directly on the tiles)
SLHA_CODEC=mixed WORK=/tmp/slha-llama integration/llama.cpp/build_and_roundtrip.sh fused

# 9. Fused-QK score diagnostics (cos / KL per layer, opt-in)
SLHA_KV_MODE=fused SLHA_FUSED_DIAG=1 SLHA_WEIGHTS_DIR=/tmp/slha-llama/weights \
  llama.cpp/build/bin/llama-perplexity -m qwen2.5-1.5b-instruct-q8_0.gguf \
  -f wiki.test.raw --chunks 2 -t 4 -fa off
```

All scripts pin the llama.cpp tag (`b9860`) and verify the commit hash before
building.

## Runtime interface

Modes are selected via environment variables:

| Variable          | Values                              |
|-------------------|-------------------------------------|
| `SLHA_KV_MODE`    | `off` / `passthrough` / `collect` / `roundtrip` / `scorediag` / `fused` |
| `SLHA_CODEC`      | `mixed` (default) / `mix3` / `grouped` / `nf4` / `tq3` |
| `SLHA_WEIGHTS_DIR`| directory with `layer-NNN.slhw` and `manifest.json` |

`fused` and `scorediag` require the standard (non-flash) attention path: the
build script passes `-fa off` for these modes because the SLHA custom node is a
GGML op on that path (flash attention would bypass it).

## Measured results

Qwen2.5-1.5B-Instruct Q8_0, WikiText-2 test, 12 chunks, 512 context, 4 threads.

| Mode        |      PPL | ΔPPL absolute | ΔPPL relative | Notes           |
| ----------- | -------: | ------------: | ------------: | --------------- |
| baseline    | 11.8753  |             — |             — | original        |
| passthrough | 11.8753  |          0.00 |          0.0% | hook sanity     |
| mixed       | 16.5976  |          4.72 |         39.8% | round-trip      |
| mix3        | 16.6460  |          4.77 |         40.2% | round-trip      |
| fused       | 19830.9  |       19819.0 |    ~167000 %  | fused-QK (this run) |

The fused row is the measurement made on this machine (aarch64, 4 threads) with
`SLHA_CODEC=mixed`, `-fa off`, 12 chunks. It is a hard NO-GO: the SLHA scores
do not preserve the attention distribution (see the score diagnostics: cos ≈
0.68, KL ≈ 5 on deep layers).

See [`results/measurements.json`](results/measurements.json) for exact SHAs,
commands, and timestamps.

## Why the regression is large

**Round-trip:** the callback reconstructs K from the SLHA tile using the latent
plus a linear estimate of the sign-LSH residual. While the latent captures the
principal subspace of the calibration K activations, the 256-bit residual
sketch is a *score-side* correction (it preserves attention dot products well
in offline score tests) rather than a faithful inverse of the quantization
error. Restoring the exact per-vector K value from sign bits alone is not what
the residual was designed for, and the perplexity measurement reflects that
limitation.

**Fused-QK:** the SLHA scores are computed directly on the tiles — no K
reconstruction — yet they still fail to replace QK^T. The score diagnostics
show why: the per-layer projections are trained on K alone (`fit_with`, the
collection seam only exposes K), so the SLHA scores are correlated with the
true QK^T (cos ≈ 0.68) but with a very different *softmax distribution* (KL ≈
5 on deep layers). Replacing the scores shifts attention mass to the wrong
tokens, which the perplexity measurement makes catastrophic. Closing the gap
would require joint K/Q calibration (collect Q at the same seam), score-scale
calibration, or a larger latent/residual budget — none validated here.

## Limitations and known issues

* Per-layer projections are trained on K only (`fit_with`), not joint K/Q,
  because the collection seam currently only exposes K — this is the most
  likely cause of the fused-QK NO-GO.
* The round-trip reconstruction uses a spectral residual estimate; it improves
  over latent-only reconstruction but is not an exact inverse.
* The fused and scorediag nodes are CPU GGML callbacks (one thread per layer);
  they run on the standard non-flash attention path only (`-fa off`).
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
