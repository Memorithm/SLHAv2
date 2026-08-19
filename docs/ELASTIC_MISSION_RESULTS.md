# Elastic Mission Results

Mission date: 2026-08-19. Baseline: `8edc1c2a` (SLHAv2 master).
Machine-readable artifact: `results/elastic_mission.json`.

Every number below is labeled **measured**, **calculated**, **simulated** or
**NOT RUN**. They are never mixed.

## Repositories

| Repo | Starting SHA | Work |
|---|---|---|
| SLHAv2 | `8edc1c2` | Full engineering pass (below) |
| CCOS Core | `b8febf7` | Read-only analysis: `MemoryProvider` boundary is intact; no code change needed (adapter is self-contained, no scirust dep) |
| CCOS Enterprise | `242dea5` (dirty tree preserved) | Read-only: governance/budget model is the correct hard-constraint layer; Elastic maps onto it without duplication |
| CCOS Research Lab | `d19b026` | Read-only: learned-policy seam identified |
| FLAT-ATTENTION | `fbcca14` | Read-only: `RuntimeDeviceCapabilities` + deterministic fingerprint already exist (name is provenance, not policy); adapter-only |

## Architecture

```text
elastic-core (no_std, zero deps)
     ↑
elastic-runtime (telemetry, journal, coordinator)
     ↑
elastic (facade + macros)
     ↑
SLHAv2 controllers (ElasticContext, ElasticKvCache — slhav2-vram)
     ↑
scirust (reference core)  ←  slhav2-vram (CUDA)  ←  slha-c / slha-python / slha-mcp
     ↑
llama.cpp shim (tile store + score hooks) → CCOS adapters
```

No cycles. The Elastic language never depends on SLHAv2.

## Elastic language

- 5 crates in `elastic/`, 46 tests, clippy -D warnings clean.
- Macros `elastic_state!` / `elastic_budget!` / `elastic_policy!` /
  `elastic_target!` lower into real runtime types; `Pinned => !Evicted`
  is a compile-time-validated negative edge.
- Independent examples: `ElasticQueue` (work queue) and a cross-resource
  `ElasticQueue + ElasticWorkers` sharing one coordinator budget — proves
  the language is not KV-cache-specific.
- **Extraction: PASS** — copied to `/tmp/elastic-standalone`, 46 tests pass,
  dependency-graph leak scan = 0 references to scirust/slha/ccos/llama.

## SLHAv2 fixes (each with its proof test)

| Defect | Fix | Proof |
|---|---|---|
| No-op PTX stub loaded as fused GEMM | PTX deleted; `GpuEngine::new` fails closed | cuda-feature build; audit |
| `cudaHostRegister` without unregister | `PinnedHostRegion` RAII | compile-time ownership |
| Tile store misalignment + pointer after unlock | 128-aligned region, documented contract | `tile_store_tests` (alignment, bounds, concurrency) |
| K tensor `ne[]/nb[]` rewritten | GGML tensor untouched; external tile store | `compile-check` vs real ggml.h |
| `SLHA_MAX_LAYERS=128` / capacity 16384 | Observed-layer sizing, fail closed | shim tests |
| Hand-written CUresult table (wrong mappings) | `cuGetErrorName`/`cuGetErrorString` | error-name tests |
| Stale GPU query bytes | exact D_C/RESIDUAL_WORDS validation | pipeline validation |
| Arena padding lost from free list | overhead accounting + conservation invariant | 5000-op random conservation test |
| Logical HOT/WARM accounting | physical residency (128/96/0 real bytes) | elastic_cache tests |
| `serde` non-optional ("zero deps" claim) | optional `serde` feature | `cargo tree` default = empty |
| MSRV 1.85 < dependency 1.89 | slhav2-vram → 1.89 | manifest |
| README "4 crates / all 0.2.0" | 5 crates + elastic incubator, versions correct | README |

## ElasticContext

Implemented in `slhav2-vram::elastic_context`. Logical context, raw KV
demand (from model topology), and physical residency are separate; the ECA
demotes before exhaustion; the model positional limit is a hard constraint.
Deterministic demo (`examples/elastic_context.rs`) shows the full lifecycle:
growth → pressure HIGH → HOT→WARM → WARM→COLD → stable, with telemetry.

Demo numbers (measured, deterministic run, demo topology 57344 B/token):

- raw KV at 4096 tokens: **224 MiB (calculated from topology)**
- physical resident high-water: **0.25 MiB (measured)**
- compression ratio: **~831× (calculated)**

## Quality gate

- baseline PPL 11.8644, strict replacement 16.8855, **Δ = +42.36%**
  (**measured**, 2026-07-29 experiment, kept as historical evidence).
- Target ≤ 1%: **NO-GO, unchanged.** The fused score replacement is not
  production-complete; the safe fallback remains available.
- No new perplexity run was possible: no model/dataset assets on this
  machine (see NOT RUN).

## CUDA

- **Compile: PASS.** **Runtime: PASS — 12/12** `--ignored` tests executed on
  the real NVIDIA Thor (driver 580.00, CUDA 13.0, nvcc 13.0.48). This is
  the first real-hardware execution of the backend; the crate README's
  "never executed" disclaimer is superseded.

## Other surfaces

- C ABI: shim suite 7/7 + compile-check against real GGML headers: PASS.
- Python: wheel built (maturin), installed, imported on aarch64: PASS.
- MCP: full lifecycle handshake + `tools/call slha.audit` 7/7 checks: PASS.

## NOT RUN

- **12-chunk perplexity re-gate** — requires Qwen2.5-1.5B-Instruct Q8_0
  GGUF + WikiText-2 slice + pinned llama.cpp b9860 build. No model weights
  or dataset on this machine; the exact reproduction command is
  `integration/llama.cpp/build_and_baseline.sh` + the strict-replacement
  patch documented in `integration/llama.cpp/README.md`.
- **Full llama.cpp roundtrip** — same assets required.
- **CI MSRV job on 1.89** — only the host 1.89 toolchain was exercised.
- **Miri** — no nightly toolchain configured.
- **Sanitizers (ASan/UBSan/TSan)** — host toolchain lacks the C++ sanitizer
  builds for the shim; recorded as not executed.

## Next research (evidence-based only)

1. **Deployable top-k-preserving score path** — the rank-transplant oracle
   proves top-16 restoration captures 98.42% of the ranking benefit; the
   next experiment is a codec/score constraint that preserves top-k ordering
   by construction without baseline-score access (recommended in the oracle
   artifact), re-measured on the 12-chunk protocol.
2. **Early-layer calibration (0–6)** — 87.8% recovery from early layers
   only; a layer-specific calibration experiment is the cheapest next
   probe.
3. **Quality-aware elasticity** — only after a validated per-tile quality
   signal exists (currently experimental; no production claim).
