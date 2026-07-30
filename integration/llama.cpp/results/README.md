# SLHA v2 × llama.cpp — Quality round-trip measurement report

## Experiment

Round-trip every K vector through an SLHA 128-byte tile before storing it in
llama.cpp's normal KV cache, then measure perplexity on WikiText-2.

## Configuration

| Item                | Value                                                       |
|---------------------|-------------------------------------------------------------|
| Model               | Qwen/Qwen2.5-1.5B-Instruct-GGUF `qwen2.5-1.5b-instruct-q8_0.gguf` |
| Evaluation data     | WikiText-2-raw-v1 `split=test` (120 kB slice)              |
| Calibration data    | WikiText-2-raw-v1 `split=train` (120 kB slice)              |
| llama.cpp tag       | `b9860`                                                     |
| llama.cpp commit    | `fdb1db877c526ec90f668eca1b858da5dba85560`                  |
| Context             | 512 tokens                                                  |
| Chunks              | 12                                                          |
| Threads             | 4                                                           |
| Insertion point     | `llama_kv_cache::cpy_k`, after `ggml_view_2d`               |
| Tensor layout       | `n_embd_gqa = 256` (full K row per token per layer)         |
| Thread safety       | per-layer mutex protects lazy model loading and dimension init |
| Activation format   | `[u32 magic=0x534C4841][u32 rows][u32 cols][f32 rows×cols LE]` |
| Weight manifest     | `manifest.json` (format_version 1, per-layer file + input_dim) |
| Codec selection     | `SLHA_CODEC` environment variable                           |

## Results

| Mode        |      PPL | ΔPPL absolute | ΔPPL relative | tok/s | Notes           |
| ----------- | -------: | ------------: | ------------: | ----: | --------------- |
| baseline    | 11.8753  |             — |             — | 44.66 | original        |
| passthrough | 11.8753  |          0.00 |          0.0% | 44.81 | hook sanity     |
| mixed       | 16.5976  |          4.72 |         39.8% | 42.28 | SLHA round-trip |
| mix3        | 16.6460  |          4.77 |         40.2% | 42.22 | SLHA round-trip |

## Conclusion

**NO-GO** for the K reconstruction quality path. The round-trip does not
preserve perplexity within the pre-registered ≤1 % gate; the regression is
≈+40 %. The sign-LSH residual preserves attention *scores* well in offline
score tests, but it is not a sufficiently accurate inverse for full K
reconstruction.

This finding does not invalidate the later fused-score path (score compressed
tiles directly), which is out of scope for this milestone.

## Checksums

* model: `d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8`
* test corpus: `136677b69515d194d28d42728ac1ba29850b67cd30715bbb7a4a023815ab01d5`
* calibration corpus: `981baada725bd8d768ceadc500e52ec4d8571f5a2904c62fbe5ad03cf6bd1293`
* weight manifest: `485031d795546a16a8a7106ac2baec7a3c4d34c8512f61e612b3ffc38e77eecb`

## Validation

* Cargo formatting, clippy, test, doc, and MSRV gates pass.
* Baseline and passthrough produce identical perplexity.
* Missing-weight and dimension-mismatch cases print explicit per-layer errors
  and fall back to passthrough for the affected layer.
* AddressSanitizer build succeeds; runtime shadow-memory allocation failed in
  the current container environment.

## Reproducibility — what this repository alone can and cannot re-derive

This section applies to every artifact in this directory
(`measurements.json`, `layerwise_score_gap.json`,
`score_temperature_calibration.json`) and to
`rank_transplant_oracle.json`, which as of 2026-07-30 is versioned on the
`research/llama-rank-transplant-oracle` branch (see
`docs/RESEARCH_BRANCHES.md`).

**Reproducible from the repository alone:**

* The complete measurement *pipeline*: shim sources, patches, build scripts
  (`build_and_baseline.sh`, `build_and_roundtrip.sh`), calibration and
  training scripts, and their C++/Rust test suites (run in CI).
* The *verification* of any re-run against the original inputs: every
  artifact pins the model, corpus, llama.cpp tag/commit and weight files by
  sha256, so a re-measurement provably uses the same inputs.

**NOT reproducible from the repository alone:**

* The measured numbers themselves. Re-deriving them requires downloading the
  pinned external inputs — `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
  (`qwen2.5-1.5b-instruct-q8_0.gguf`, sha256 `d7efb072…`) and
  WikiText-2-raw-v1 slices (sha256 pinned above) — building llama.cpp at tag
  `b9860`, and hours of CPU time. The repository stores the *evidence and the
  recipe*, not the inputs.
* The completeness-gate transcripts of `rank_transplant_oracle.json`
  (screening 75/75, twelve-chunk 27/27, determinism runs). They were written
  to an ephemeral session container scratch directory that no longer exists.
  **Only their sha256 values survive**, recorded inside the artifact itself
  (`frozen_gates.gate_artifacts`). They cannot be reconstructed, and this
  README does not pretend otherwise: a future re-run can regenerate
  *equivalent* transcripts and gates, but not the original byte streams the
  recorded hashes describe.
