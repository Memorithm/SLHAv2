# SLHAv2 repository agent instructions

Before making repository changes, fetch and read the persistent off-main ecosystem roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/SLHAV2_ECOSYSTEM_ROADMAP.yaml
```

Treat the repository-root `AGENTS.md` as useful local workspace intelligence, but treat the off-main roadmap as authoritative for current strategy, real-model milestones, ecosystem ownership, and promotion state.

Reread the roadmap at every session start, before selecting the next real-model or integration phase, before ElasticXxx/NNIS/FLAT-ATTENTION/Forge/Hub/Verify/SciCapsule integration, after strategy or quality-budget changes, and before PR or merge decisions affecting model integration, KV format, quality, or performance.

If the roadmap is unavailable, fail closed for major model-integration, KV-format, quality-promotion, performance-promotion, cross-repository integration, or merge decisions. Do not substitute guesses for missing roadmap state.

Proxy benchmarks are not real-model evidence. Real-model claims require the same model, tokenizer, prompt/workload, decode semantics, environment scope, and an executable baseline path.
