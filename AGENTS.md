# SLHAv2 Agent Bootstrap Contract

Before any autonomous coding, real-model integration, KV-format change, performance/quality promotion, cross-repository integration, PR creation, or merge decision, fetch and read the persistent roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/SLHAV2_ECOSYSTEM_ROADMAP.yaml
```

For any real-model, KV execution, codec, quality, memory, throughput, attention-adapter, Elastic, or cross-repository ML work, also read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

The off-main roadmap is authoritative for current strategy, ecosystem ownership, real-model milestones, and promotion state. The ML maturity overlay makes 5/5 an evidence-backed exit criterion: proxy decode, tile-size projections, codec SNR, or synthetic kernel fidelity never substitute for a real tokenizer + real weights + prefill + autoregressive decode comparison against the same uncompressed baseline.

If either required roadmap or the applicable ML overlay cannot be fetched or read, fail closed for major model-integration, KV-format, quality-promotion, performance-promotion, cross-repository integration, or merge decisions. Read-only diagnosis is allowed.

Reread them at every session start, before selecting the next major phase, before integrating ElasticXxx/NNIS/FLAT-ATTENTION/TurboQuant/Forge/Hub/Verify/SciCapsule, after strategy or quality-budget changes, and before relevant PR/merge decisions.

Do not merge the roadmap or ML maturity overlay itself into the default branch unless the user explicitly requests it.

The workspace intelligence below remains useful as a local implementation snapshot, but verify it against the current head, persistent roadmap, and ML maturity overlay before relying on milestone or CI-status statements.

---

# AGENTS.md — SLHAv2 Workspace Intelligence

## Project
SLHAv2 — sparse linear algebra attention scoring for KV-cache eviction decisions, with CPU reference (`scirust`) and CUDA-accelerated (`slhav2-vram`) backends.

## Current Milestone
**feat/elastic-architecture-v1** — Elastic resource language incubator + physical residency + P0 remediation (see `docs/ELASTIC_MISSION_RESULTS.md`). The fused-score quality gate remains **NO-GO** (ΔPPL +42.36% vs ≤1% target); everything else in the mission is engineering-complete and gate-green.

## Architecture
- `elastic/` — **Elastic resource language incubator** (5 crates, zero SLHA dependencies; extraction-proven)
  - `elastic-core/` — generic ECA: traits, pressure, hysteresis, tiers, budgets, transactions, forecasts, decision traces (`no_std`, zero deps, no unsafe)
  - `elastic-runtime/` — telemetry, deterministic journal, `ElasticCoordinator`
  - `elastic-macros/` — `elastic_state!`/`elastic_budget!`/`elastic_policy!`/`elastic_target!` with real semantics
  - `elastic-testkit/` — `ScriptedBackend` fault injection + `ElasticSimulator`
  - `elastic/` — facade + prelude + independent examples (ElasticQueue/Workers)
- `scirust/` — authoritative source of truth (CPU reference, audit, metrics, JSON); default build is zero-dependency (`serde` is an optional feature)
- `slhav2-vram/` — CUDA-accelerated implementation + **physical** `ElasticKvCache` + `ElasticContext`
  - `src/elastic_cache.rs` — physically-resident cache (HOT 128 / WARM 96 packed / COLD 0 real bytes) on the generic ECA
  - `src/elastic_context.rs` — runtime context-resource controller (topology-driven KV cost, predictive demotion, positional-limit hard constraint, telemetry)
  - `src/codec.rs` — constants, NF4 codebook, safe byte helpers, scoring math, all 6 codecs (INT4/NF4/MIXED/TQ3/MIX3) + `validate_codec`, WARM pack/unpack
  - `src/mem/arena.rs` — 128-offset-aligned arena with **conservation accounting** (used + free + overhead == capacity)
  - `src/backends/cuda.rs` — CUDA Driver API engine; real `cuGetErrorName`/`cuGetErrorString` error reporting
  - `src/mem/tile.rs` — `SerializedTile` (128B), `SerializedSlice`
  - `src/pipeline.rs` — `score_tiles_cpu`, `copy_scores_from_gpu`, `GpuScoringPipeline` (exact-length query validation)
  - `kernels/slha_score.cu` — CUDA kernel (all 6 codecs, `__constant__` NF4 codebook) — **runtime-validated on NVIDIA Thor (12/12)**
- `integration/llama.cpp/` — shim: external 128-aligned tile store (no GGML tensor mutation), observed-layer sizing, codec fail-closed

## Key Decisions
- All six scirust codecs implemented: uniform signed INT4 (zero-point `nibble - 8`), NF4, MIXED, TQ3, MIX3 (with `FLAG_TQ3_NOCORR`)
- Unknown flag combinations are rejected (`validate_codec`) instead of silently decoded as INT4
- `LATENT_KV_WORDS` → `LATENT_BYTES` (re-exported from scirust, value 64)
- Tile field offsets: latent 0..64, residual 64..96, scale 96(4B), lambda 100(4B), flags 118(2B), group_scales 120..128
- CUDA: opaque zero-sized struct handles, `CUdeviceptr` = `u64`, all allocations share `Rc<CudaInner>`, engine is `!Send`/`!Sync`; errors via `cuGetErrorName`/`cuGetErrorString`
- Arena: offsets aligned to 128 (tile natural alignment), real sizes kept — 128B tiles pack at 128B stride (2× capacity vs the old 256B rounding); O(log n) free/alloc via BTreeMap; **conservation invariant** (used + free + overhead == capacity)
- **Elastic direction**: `elastic-core ← elastic-runtime ← elastic ← SLHAv2 ← integrations`; nothing in `elastic/` imports SLHAv2 (extraction-proven)
- **Physical residency**: WARM = 96B packed form (`codec::pack_warm`); reported bytes == allocated bytes
- **No arbitrary caps**: llama shim sizes layers from the observed model; unknown `SLHA_CODEC` fails closed
- The GGML K tensor is never reshaped; SLHA tiles live in the external aligned tile store
- The stub `fused_gemm.ptx` is deleted; hybrid GPU path is quarantined (fails closed), production CUDA is `slhav2-vram`
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
# Elastic standalone (extraction readiness)
(cd elastic && cargo test --workspace) && cargo tree --workspace | grep -ciE 'scirust|slha|ccos|llama' # expect 0
# CUDA tests (requires hardware — validated on NVIDIA Thor, 12/12)
SLHAV2_REQUIRE_CUDA=1 cargo test -p slhav2-vram --features cuda -- --ignored --nocapture --test-threads=1
# llama.cpp shim tests + compile-check against real ggml headers
(cd integration/llama.cpp/tests && make test && make compile-check LLAMA_CPP_DIR=/path/to/llama.cpp)
```

## CI Status (historical snapshot; verify current head before relying on this section)
- Gate A: Formatting was reported passing at the snapshot that created this file.
- Gate B: Clippy was reported passing at that snapshot.
- Gate C: workspace tests were reported green at that snapshot; CUDA tests were compile-only in CI and separately hardware-validated on NVIDIA Thor during that mission.
- Elastic standalone extraction, llama shim, Python wheel, and MCP were reported validated at that snapshot.
