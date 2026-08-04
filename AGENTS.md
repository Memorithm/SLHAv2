# AGENTS.md — SLHAv2 Workspace Intelligence

## Project
SLHAv2 — sparse linear algebra attention scoring for KV-cache eviction decisions, with CPU reference (`scirust`) and CUDA-accelerated (`slhav2-vram`) backends.

## Current Milestone
**fix/slhav2-vram-cuda-repair** — Repair `slhav2-vram` CUDA and codec implementation against `scirust` as exact source of truth.

## Architecture
- `scirust/` — authoritative source of truth (CPU reference, audit, metrics, JSON)
- `slhav2-vram/` — CUDA-accelerated implementation
  - `src/codec.rs` — constants, NF4 codebook, safe byte helpers, scoring math, all 6 codecs (INT4/NF4/MIXED/TQ3/MIX3) + `validate_codec`
  - `src/mem/tile.rs` — `SerializedTile` (128B), `SerializedSlice`
  - `src/mem/arena.rs` — 128B-offset-aligned arena (BTreeMap free list, O(log n) alloc/free/coalesce)
  - `src/backends/cpu.rs` — CPU scoring engine
  - `src/backends/cuda.rs` — CUDA Driver API engine (opaque handles, `Rc<CudaInner>`, `CudaFunction` owns its module)
  - `src/pipeline.rs` — `score_tiles_cpu`, `copy_scores_from_gpu`, `GpuScoringPipeline` (persistent arena + streams)
  - `kernels/slha_score.cu` — CUDA kernel (all 6 codecs, `__constant__` NF4 codebook)
  - `tests/cpu.rs` — scirust parity tests (all codecs + rejection)
  - `tests/cuda.rs` — GPU validation (`#[ignore]`, `SLHAV2_REQUIRE_CUDA=1`)
  - `examples/cuda_validation.rs` — standalone validation
  - `benches/score.rs` — CPU benchmark

## Key Decisions
- All six scirust codecs implemented: uniform signed INT4 (zero-point `nibble - 8`), NF4, MIXED, TQ3, MIX3 (with `FLAG_TQ3_NOCORR`)
- Unknown flag combinations are rejected (`validate_codec`) instead of silently decoded as INT4
- `LATENT_KV_WORDS` → `LATENT_BYTES` (re-exported from scirust, value 64)
- Tile field offsets: latent 0..64, residual 64..96, scale 96(4B), lambda 100(4B), flags 118(2B), group_scales 120..128
- CUDA: opaque zero-sized struct handles, `CUdeviceptr` = `u64`, all allocations share `Rc<CudaInner>`, engine is `!Send`/`!Sync`
- Arena: offsets aligned to 128 (tile natural alignment), real sizes kept — 128B tiles pack at 128B stride (2× capacity vs the old 256B rounding); O(log n) free/alloc via BTreeMap, no per-free full sort
- Build script: fails hard on nvcc error, emits PTX under `OUT_DIR`
- `util.rs` deleted; safe byte helpers in `codec.rs`
- No `bytemuck` dependency

## Gate Commands
```bash
# Gate A — formatting
cargo fmt --all -- --check
# Gate B — clippy (all targets, all features)
cargo clippy --workspace --all-targets --all-features -- -D warnings
# Gate C — tests (workspace)
cargo test --workspace --all-features
# CUDA tests (requires hardware)
SLHAV2_REQUIRE_CUDA=1 cargo test --features cuda -- --ignored --nocapture --test-threads=1
```

## CI Status (current commit)
- ✅ Gate A: Formatting passes
- ✅ Gate B: Clippy passes (0 warnings)
- ✅ Gate C: workspace test suite green (`cargo test --workspace --all-features` — ~280 tests incl. 124 scirust lib, 6 scirust integration files, 24 slha-c, 27 slhav2-vram lib + 20 CPU integration; CUDA tests are `#[ignore]`d and compile-only in CI)
