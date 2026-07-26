# AGENTS.md — SLHAv2 Workspace Intelligence

## Project
SLHAv2 — sparse linear algebra attention scoring for KV-cache eviction decisions, with CPU reference (`scirust`) and CUDA-accelerated (`slhav2-vram`) backends.

## Current Milestone
**fix/slhav2-vram-cuda-repair** — Repair `slhav2-vram` CUDA and codec implementation against `scirust` as exact source of truth.

## Architecture
- `scirust/` — authoritative source of truth (CPU reference, audit, metrics, JSON)
- `slhav2-vram/` — CUDA-accelerated implementation
  - `src/codec.rs` — constants, NF4 codebook, safe byte helpers, scoring math
  - `src/mem/tile.rs` — `SerializedTile` (128B), `SerializedSlice`
  - `src/mem/arena.rs` — 256B-aligned bump arena
  - `src/backends/cpu.rs` — CPU scoring engine
  - `src/backends/cuda.rs` — CUDA Driver API engine (opaque handles, `Rc<CudaInner>`)
  - `src/pipeline.rs` — `score_tiles_cpu`, `copy_scores_from_gpu`
  - `kernels/slha_score.cu` — CUDA kernel (INT4/NF4 only)
  - `tests/cpu.rs` — scirust parity tests (11 tests)
  - `tests/cuda.rs` — GPU validation (`#[ignore]`, `SLHAV2_REQUIRE_CUDA=1`)
  - `examples/cuda_validation.rs` — standalone validation
  - `benches/score.rs` — CPU benchmark

## Key Decisions
- Only two codecs: uniform signed INT4 (zero-point `nibble - 8`) and NF4 (scirust-identical codebook)
- No MIXED/TQ3/MIX3/TQ3_NOCORR flags or codecs (removed to match scirust)
- `LATENT_KV_WORDS` → `LATENT_BYTES` (re-exported from scirust, value 64)
- Tile field offsets: latent 0..64, residual 64..96, scale 96(4B), lambda 100(4B), flags 118(2B), group_scales 120..128
- CUDA: opaque zero-sized struct handles, `CUdeviceptr` = `u64`, all allocations share `Rc<CudaInner>`, engine is `!Send`/`!Sync`
- Arena: checked 256-byte alignment (`(value + 255) & !255`), not `next_power_of_two`
- Build script: fails hard on nvcc error, emits PTX under `OUT_DIR`
- `util.rs` deleted; safe byte helpers in `codec.rs`
- No `bytemuck` dependency

## Gate Commands
```bash
# Gate A — formatting
cargo fmt --all -- --check
# Gate B — clippy (no default features)
cargo clippy --all-targets --no-default-features
# Gate C — tests (no default features)
cargo test -p slhav2-vram --no-default-features
# CUDA tests
SLHAV2_REQUIRE_CUDA=1 cargo test --features cuda -- --ignored --nocapture --test-threads=1
```

## CI Status (current commit)
- ✅ Gate A: Formatting passes
- ✅ Gate B: Clippy passes (0 warnings)
- ✅ Gate C: 38 tests pass (27 lib + 11 CPU integration)
