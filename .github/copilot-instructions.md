# SLHAv2 repository agent instructions

Before making repository changes, fetch and read the persistent off-main ecosystem roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/SLHAV2_ECOSYSTEM_ROADMAP.yaml
```

For real-model, KV execution, codec, quality, memory, throughput, attention-adapter, Elastic, or cross-repository ML work, also read:

```bash
git show origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

Treat the repository-root `AGENTS.md` as mandatory bootstrap policy and local workspace intelligence, and treat the off-main roadmap plus applicable ML overlay as authoritative for current strategy, real-model milestones, ecosystem ownership, promotion state, and maturity gates.

Reread them at every session start, before selecting the next real-model or integration phase, before ElasticXxx/NNIS/FLAT-ATTENTION/TurboQuant/Forge/Hub/Verify/SciCapsule integration, after strategy or quality-budget changes, and before PR or merge decisions affecting model integration, KV format, quality, or performance.

If the roadmap or applicable ML overlay is unavailable, fail closed for major model-integration, KV-format, quality-promotion, performance-promotion, cross-repository integration, or merge decisions. Do not substitute guesses for missing roadmap state.

Proxy benchmarks are not real-model evidence. Real-model claims require the same model, tokenizer, prompt/workload, decode semantics, environment scope, and an executable baseline path. A `5/5` label additionally requires the overlay's end-to-end quality, memory, throughput, interoperability and exact-head evidence gates.
