# ADR 0001 — llama.cpp as the first real-model reference engine

- Status: Accepted for the first real-model integration
- Date: 2026-08-29
- Repository base: `master` at `6df70781e7d216c5b30e5e7b1001a865fa1f8f3f`
- Reference engine: llama.cpp tag `b9860` (`fdb1db877c526ec90f668eca1b858da5dba85560`), matching the existing integration scripts

## Context

SLHAv2 already contains an engine-independent Rust core, a stable C ABI, Python bindings, CCOS/elastic cache mechanisms, and an existing llama.cpp shim. The llama.cpp path has already exercised real tokenization, model loading, prefill/decode and perplexity experiments on a real GGUF model. Ollama and vLLM material in the repository is integration-design material rather than an equally mature code path.

The remaining product/scientific milestone is not another synthetic attention benchmark. It is to make a real engine run without a full context-sized persistent K tensor while preserving the ordinary V cache, tokenization, model graph, masks, RoPE-applied current K generation, autoregressive sampling and observable engine lifecycle.

## Decision

Use **llama.cpp only** for the first reference integration.

The engine-specific implementation remains under `integration/llama.cpp/`. The Rust `scirust` core and its C ABI remain engine-independent. No llama.cpp dependency is added to the default Cargo workspace graph.

The first physical-K mode is deliberately separate from the existing diagnostic `tilestore + score replacement` path. It is opt-in and fail-closed. It may reuse the existing SLHA tile format, weights, encoder and scorer, but it must not claim memory savings while llama.cpp still owns a context-sized K tensor.

## Required invariants for physical-K mode

1. No context-sized GGML K allocation. A constant-size metadata sentinel is acceptable only if it is measured and reported as runtime overhead, not counted as compressed KV.
2. V remains owned by llama.cpp for the first integration.
3. Current K is captured after the model's normal RoPE/current-token transformations at the existing KV-write seam, then encoded into the SLHA tile store.
4. Attention logits are produced from SLHA tiles without first materializing baseline `Q*K`.
5. FlashAttention is rejected in this mode until an explicit compressed-K kernel exists.
6. Diagnostic modes that require paired baseline logits (score oracle, rank metrics, offline scale fitting/training labels) are rejected in this mode.
7. The current score-replacement implementation supports one KV stream; physical-K mode therefore rejects multi-stream use rather than silently changing semantics.
8. K-shift/context-rotation operations are unsupported until the compressed representation can be transformed correctly. They must fail rather than using stale RoPE state.
9. Reset must clear the external tile store. Existing llama.cpp cell metadata remains authoritative for masking and sequence occupancy.
10. State save/load of a physical-K context is not accepted as correct until compressed K serialization is implemented and tested.

## Why llama.cpp first

- The repository already pins and builds llama.cpp and has a C++ shim at the exact K write and attention-score seams.
- Existing real-model measurements make regressions comparable with earlier work.
- The existing C ABI provides projection loading, K encoding, prepared-query construction and tile scoring without putting engine dependencies into `scirust`.
- llama.cpp exposes KV lifecycle/cell metadata sufficiently close to the integration seam to test reset, append, capacity and sequence semantics.
- Maintaining one engine integration first avoids duplicating unvalidated behavior across Ollama/vLLM adapters.

## Explicit non-goals of PR 1

PR 1 does not claim that CCOS HOT/WARM/COLD is wired into llama.cpp; that belongs to the later real-model CCOS PR after the physical K path is validated. It also does not claim speedup, memory reduction, perplexity, token agreement, or fidelity until the corresponding real-model commands have been executed and their provenance recorded.

## Acceptance evidence

A later benchmark/report commit must identify the exact GGUF hash, quantization, llama.cpp revision, SLHAv2 commit, host/CPU/RAM/GPU, context, thread count and command line. Baseline and physical-SLHA runs must use the same model and workload. Process RSS and engine/KV allocations must be measured separately; no whole-process saving may be inferred solely from the 128-byte tile size.
