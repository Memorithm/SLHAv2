# Elastic Mission Audit — SLHAv2 current-state ground truth

Date: 2026-08-19. Baseline: `8edc1c2a6cd744e924f7818fd21e02557fb8bc4d`
(branch `master`; mission branch `feat/elastic-architecture-v1`). All findings
verified against the working tree at that commit. Host: NVIDIA Thor (CUDA 13.0,
nvcc 13.0.48, driver 580.00), rustc 1.89.0, cmake 3.28.3, `/root/llama.cpp`.

Classification: **P0** correctness/safety · **P0-S** scientific validity ·
**P0-E** elastic foundation · **P1** architecture/perf/integration · **P2**
ergonomics/docs.

Legend: VERIFIED = defect confirmed in source; OBSOLETE = no longer present;
REMEDIATED = fixed during this mission (link to proof).

---

## P0 — correctness / safety

| # | File | Defect | Why it matters | Remedy | Test |
|---|---|---|---|---|---|
| P0-1 | `scirust/kernels/fused_gemm.ptx` | `.visible .entry stub` — a no-op kernel that just `ret`s. `GpuEngine::new` loads it as production "fused GEMM"; `launch_kernel` launches the stub. | A no-op kernel must never satisfy a production success criterion. | Remove the stub path; route GPU through the canonical real backend (`slhav2-vram` CUDA engine). Quarantine the fake PTX. | Removed file; workspace builds without it; CUDA real-kernel tests pass |
| P0-2 | `scirust/src/engine/hybrid.rs` | `cudaHostRegister` without any `cudaHostUnregister` (no RAII owner); pinned memory may be moved/freed while registered. Also 1 MiB fixed arena, weak dim validation, zero-dim division risk. | Pinned-registration lifetime/use-after-free class bug; UB risk. | Quarantine experimental hybrid path behind explicit experimental gating; `register_pinned_memory` becomes an owning RAII pinned-region type, or the path is removed from production. | Compile gate; documented experimental |
| P0-3 | `integration/llama.cpp/shim/slha_llama.cpp` `slha_tile_store::read` | Returns `tiles.data() + ...` as `const void*` after releasing the mutex; caller casts to `const SciRustSlhaTile*`. `std::vector<std::byte>` has no alignment guarantee. | Use-after-unlock, data race with `write`/`clear_layer`/`reset`, misaligned tile reads. | Aligned typed storage (`alignas(128)` tile array), guarded access, generation counter, or copy-out semantics; document pointer lifetime. | Tile store tests (alignment, concurrent write/read, reset invalidation) |
| P0-4 | `integration/llama.cpp/shim/slha_llama.cpp` `slha_intercept_k_cache_allocation` | Rewrites `k_tensor->ne[]/nb[]` to pretend a GGML K tensor is an I8 byte buffer (`reduced_bytes = n_tokens * 128`), after the tensor was allocated with the real `type_k`. | Tensor metadata rewriting does not prove physical allocation semantics; `sizeof(float)` doesn't describe arbitrary GGML K types; other llama code may assume original shape/type. | External SLHA-owned storage backend for the physical K store; never rewrite the GGML tensor shape. | Roundtrip integration test; no `ne[]` mutation |
| P0-5 | `integration/llama.cpp/shim/slha_replace_counters.hpp` `#define SLHA_MAX_LAYERS 128`; `slha_llama.cpp` `state.num_layers = 128`, `capacity = 16384` | Arbitrary model-derived caps. A model with >128 layers or >16384 positions silently truncates/overflows (tile store write fails with a printed error, or counters are out of range). | Hidden truncation forbidden. | Derive layers from the runtime-observed layer count; size stores dynamically or fail closed. | Layer-cap test with observed count > caps |
| P0-6 | `slhav2-vram/src/backends/cuda.rs` `cu_result_name` | Hand-written CUDA error table is incomplete and partially wrong (e.g. 205 listed as `CUDA_ERROR_OUT_OF_MEMORY` — actual 205 is `CUDA_ERROR_INVALID_CONTEXT`; 700/701/702 vs 708/709/710 duplicated; several codes wrong). | Incorrect diagnostics; masked real errors. | Use `cuGetErrorName`/`cuGetErrorString` behind safe wrappers (fallback to verified table). | Error-name tests incl. codes 2, 205, 700, 715 |
| P0-7 | `slhav2-vram/src/pipeline.rs` `GpuScoringPipeline::score_into` | Accepts arbitrary `q_coarse`/`q_sign` lengths; copies whatever length is given into persistent device buffers, leaving stale bytes from previous queries when the new query is shorter. Kernel assumes exact D_C/RESIDUAL_WORDS. | Stale GPU query data; wrong scores. | Exact typed query validation (D_C f32s, RESIDUAL_WORDS u64s) before launch. | Stale-buffer test (short query after long query fails) |
| P0-8 | `slhav2-vram/src/mem/arena.rs` `DeviceArena::allocate` | Aligns offset to 128 but never reinserts `[old_offset, aligned_offset)` into the free list; also `used` doesn't count padding. `free_bytes()` = capacity − used claims more free than exists. | Bytes lost from accounting; `free_bytes` lies; fragmentation grows. | Reinsert prefix padding into free list; track `used` including alignment overhead; expose `overhead_bytes()`. | Conservation test: used + free + overhead == capacity |
| P0-9 | `scirust/src/attention/slha_v2/constants.rs` `LATENT_KV_WORDS` naming | Old name re-exported as `LATENT_BYTES` — documented in AGENTS.md as resolved; verify. | Naming consistency. | (verify only) | — |
| P0-10 | `slha-c/src/lib.rs` `pointer_is_aligned` | `#[allow(clippy::manual_is_multiple_of)]` — lint unknown to the pinned clippy → `-D unknown-lints` failure. | CI clippy gate broken on this toolchain. | `#[allow(unknown_lints, clippy::manual_is_multiple_of)]`. | Clippy gate passes |
| P0-11 | `slha-c/src/lib.rs`, `slhav2-vram/src/backends/cuda.rs`, `scirust/src/attention/slha_v2/tests.rs` | Unsafe blocks missing `// SAFETY:` preconditions (13 in cuda.rs) → `-D warnings` gate broken. | Gate failure. | Document every unsafe block. | Clippy gate passes |

## P0 — scientific validity

| # | File | Defect | Why it matters | Remedy | Test |
|---|---|---|---|---|---|
| P0-S1 | `integration/llama.cpp/results/measurements.json` | Registered quality gate F: **relative ΔPPL = +42.36%** (Qwen2.5-1.5B-Instruct Q8_0, WikiText-2, 12 chunks; pass-through 11.8644 → strict replacement 16.8855). Target ≤ 1%. | NO-GO on the pre-registered criterion. | Do not move goalposts. Keep fused replacement NO-GO until a deployable fix passes; research only. | Perplexity gate re-run |
| P0-S2 | `integration/llama.cpp/results/rank_transplant_oracle.json` | Oracle A (baseline ranking + SLHA values) recovers 65.92% of the gap; top-16 restoration ≈ 98.42% of that benefit; early layers 0–6 recover 87.80%. Not deployable (reads baseline scores). | Confirms key-ranking is the dominant mechanism; order-preserving residual 34.08% remains. | Ranking-preserving training objectives (existing `ranking` module); layer-specific calibration; top-k-preserving score path. | New experiments preserve lineage |
| P0-S3 | `docs/SUCCESS_CRITERIA.md`, `PLAN.md` | Honest NO-GO documentation exists — preserve it. | Do not erase inconvenient evidence. | Keep; supersede only with newer measurements marked historical. | — |
| P0-S4 | K-cache vs full KV | Documentation must state exactly what is compressed: the llama integration compresses the **K** side (tile store) only; V is untouched. `README` wording "KV-cache" must be audited. | Marketing shorthand may overstate physical implementation. | Document K-only compression explicitly; keep "KV" only where both are handled. | README audit |

## P0 — elastic foundation

| # | File | Defect | Why it matters | Remedy | Test |
|---|---|---|---|---|---|
| P0-E1 | `scirust/src/ccos.rs` `ElasticKvCache` | `live_bytes()` is *logical* accounting (HOT 128 / WARM 96 / COLD 0); tiles physically remain 128 B in the arena `Vec`. Doc comment admits it. | Reported bytes don't correspond to physically allocated bytes; cannot claim "RAM saved". | Physical residency redesign: HOT = full 128B slot; WARM = physically packed 96B (or shared) representation; COLD = released storage. | Physical bytes == allocated bytes test |
| P0-E2 | No ElasticContext | No runtime context-resource controller exists; context size is a configuration constant. | The mission's central doctrine is unimplemented. | Implement `ElasticContext` on the new generic engine. | ElasticContext tests |
| P0-E3 | No generic Elastic engine | No reusable ECA (observe/model/predict/optimize/act/verify/learn), no budget hierarchy, no hysteresis, no transactional transitions shared across resources. | Every project reinvents elasticity; no deterministic control core. | New `elastic/` incubator crates (core/runtime/macros/testkit/facade). | elastic test suite |

## P1 — architecture / performance / integration

| # | File | Defect | Remedy |
|---|---|---|---|
| P1-1 | `scirust/Cargo.toml` | Claims "zero external dependencies by default" but `serde`/`serde_json` are non-optional. | Make them optional (feature `serde`), or drop the claim. Prefer a clean lightweight core. |
| P1-2 | `slhav2-vram/Cargo.toml` | `rust-version = 1.85` while depending on scirust `rust-version = 1.89`. Effective MSRV cannot be lower. | Coherent workspace MSRV (1.89), enforced in CI. |
| P1-3 | `README.md` | Claims "4 crates" while workspace has 5 members; claims all v0.2.0 while slhav2-vram is 0.1.0. | Correct documentation. |
| P1-4 | GPU duplication | `scirust::engine::hybrid` (cudarc) vs `slhav2-vram` (CUDA driver API). Two GPU directions. | Canonical: slhav2-vram is the production CUDA backend; hybrid stays experimental/gated. |
| P1-5 | llama.cpp flash attention | Fused integrations may depend on standard attention path; flash-attention status undocumented. | Audit and document; keep honest. |
| P1-6 | `slha_codec_from_env` | Unknown `SLHA_CODEC` prints and falls back to MIXED — permissive fallback in a scientific path. | Fail closed in reproducible mode (env `SLHA_STRICT=1` or default strict); document permissive mode. |
| P1-7 | `ElasticKvCache` policy | Single `enforce_budget` ladder; no hysteresis; no forecast; no decision trace. | Rebuild on generic engine with ECA semantics. |

## P2 — ergonomics / docs / future

| # | File | Defect | Remedy |
|---|---|---|---|
| P2-1 | `docs/` | No Elastic doctrine documents exist. | Add ELASTIC_DOCTRINE, CONTROL_ALGORITHM, RUST_LANGUAGE, SAFETY_INVARIANTS, EXTRACTION_PLAN, ADOPTION_MATRIX. |
| P2-2 | `AGENTS.md` | Needs architecture update after mission. | Update. |
| P2-3 | MSRV docs | rust-version claims need verification against `cargo msrv`-style check. | Test declared MSRV in CI. |
| P2-4 | `install.sh`, `scripts/` | Audit for shell-quoting/temp-dir/checksum issues (historical hardening exists; verify preserved). | Re-audit; preserve supply-chain gates. |

## Verified-obsolete findings

- P0-9 (LATENT_KV_WORDS): re-exported alias exists; no functional defect found. VERIFIED OBSOLETE as a defect.
- None of the "hardened C handle registry, ABA/double-release protections, Python aligned tile ownership, installed-wheel runtime tests" were found regressed. VERIFIED PRESERVED.

## Remedy test inventory (planned)

- Arena conservation property test (random sequences, fragmentation, alignment).
- Stale GPU query rejection test.
- Tile-store alignment + concurrency test.
- Layer-cap dynamic sizing test.
- CUDA error-name verification test.
- Elastic crates standalone extraction test.
