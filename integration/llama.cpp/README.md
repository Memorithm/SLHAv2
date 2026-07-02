# SLHA v2 × llama.cpp — Phase 2 integration

Status: **staged, honest partial.** This directory holds the reproducible
baseline and the concrete patch plan for measuring SLHA's end-to-end quality
cost on a real LLM (PLAN.md Phase 2). What is done here is real and tested;
what is not is stated plainly.

## Scope (read this first)

The goal of this phase is the **quality-path GO/NO-GO**: round-trip every K
vector through an SLHA tile (encode → decode) *before* it is stored in
llama.cpp's KV cache, leaving the attention math untouched, and measure the
change in perplexity vs baseline. This isolates the quality cost of
SLHA-compressing K on a real model.

It is **not** the fused-score bandwidth win (scoring the compressed tile
without decompressing). That is a separate, later phase; the quality path
answers "does the compression preserve the model?" first.

## What is done (real, in this repo)

1. **The C ABI bridge** an engine needs — implemented and tested in `slha-c`
   (`slha-c/src/lib.rs`, header `slha-c/include/slha.h`):
   - `slha_weights_load(path)` / `slha_weights_free(model)` — load a `.slhw`
     projection (one per layer);
   - `size_t slha_model_dim(model)` — the projection input dim `d`;
   - `slha_encode_key(model, key, d, pos, codec, out_tile)` — encode a
     `d`-dim K vector into a 128-byte tile (`codec`: 0 int4-single, 1
     int4-grouped, 2 nf4, 3 mixed, 4 tq3, 5 mix3);
   - `slha_decode_latent(model, tile, out, d)` — reconstruct the tile's latent
     back into the original `d`-dim space.
   Panic-free (NULL checks + `catch_unwind`), round-trip tested end to end
   (`cargo test -p slha-c`).

2. **A reproducible baseline** — `build_and_baseline.sh` clones llama.cpp at a
   pinned tag, builds it CPU-only, fetches a small model + a WikiText-2 slice,
   builds `libslha`, and runs `llama-perplexity`.

   Measured baseline (Qwen2.5-0.5B-Instruct Q8_0, WikiText-2 test, 12 chunks,
   4 threads, llama.cpp tag **b9860**):

   | run | PPL |
   |---|---|
   | baseline (no SLHA) | **17.72 ± 0.96** |

## What is not done (the staged patch)

The K-cache interception is **not yet applied**. The remaining work is now
concretely specified against this llama.cpp version:

### The patch seam

The K vectors are written to the cache in
`src/llama-kv-cache.cpp`, function `llama_kv_cache::cpy_k(ctx, k_cur, k_idxs,
il, sinfo)` (~line 1311). It builds a ggml graph node:

```cpp
k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);
...
return ggml_set_rows(ctx, k, k_cur, k_idxs);   // <- k_cur written here
```

`cpy_k` is graph-building, not eager, so the SLHA round-trip must be a graph
node too. The clean insertion is a `ggml_map_custom1` between the view and
`ggml_set_rows`:

```cpp
// libslha: one projection per layer, trained offline (see below)
k_cur = ggml_map_custom1(ctx, k_cur, slha_roundtrip_k, GGML_N_TASKS_MAX,
                         g_slha_layer[il]);
return ggml_set_rows(ctx, k, k_cur, k_idxs);
```

`slha_roundtrip_k` is a small C++ shim (CPU): for each column (one token's
`n_embd_gqa`-dim K vector) it calls `slha_encode_key` then
`slha_decode_latent` via `libslha`, writing the reconstructed vector back in
place. Attention downstream is unchanged; only the stored K is the
SLHA-reconstructed K.

### The remaining tasks

1. **Per-layer projections.** SLHA needs `d > D_C = 128`; a per-head
   `head_dim` (64–128) is too small, so operate on the **full K row**
   (`n_embd_gqa`, e.g. 896 for Qwen-0.5B) as one `d`-dim vector per token per
   layer, with **one `.slhw` projection per layer**. Train them offline: a
   calibration pass dumps each layer's K activations (via a `map_custom` in
   collect-only mode, or `llama-eval-callback`), then `scirust`'s
   `train_on_real_activations` produces one projection per layer.
2. **The shim + build glue.** Add `slha_roundtrip_k`, a global
   `g_slha_layer[]` of loaded models (env-gated so a normal build is
   untouched), and link `libslha.a`.
3. **Measure.** Re-run the baseline perplexity command with the shim active
   for `{passthrough (sanity, Δ≈0), mixed, mix3}` and record Δppl + tok/s.

The offline evidence already bounds what to expect: on real GPT-2 c6
activations the best codec (mixed/mix3) reconstructs K at attention-output
cosine 0.984 but KL 0.055 — above the strict pre-registered gate (see
`docs/TURBOQUANT.md` §3bis/§3ter/§3quater). Whether that 0.055 KL actually
moves perplexity on a real model is exactly the number this patch produces.

## Reproduce the baseline

```bash
# from the SLHAv2 repo root
cargo build --release -p slha-c
WORK=/tmp/slha-llama integration/llama.cpp/build_and_baseline.sh
```
