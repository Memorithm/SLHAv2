# Elastic Extraction Plan

The `elastic/` directory inside SLHAv2 is a cleanly isolated incubator that
must be extractable into a dedicated repository with minimal or zero
redesign.

## Current state

```
elastic/
  Cargo.toml            # workspace: elastic-core, elastic-runtime,
                        #           elastic-macros, elastic-testkit, elastic
  elastic-core/         # no_std + alloc, ZERO dependencies, no unsafe
  elastic-runtime/      # std runtime: telemetry, journal, coordinator
  elastic-macros/       # proc macros (proc-macro2, quote, syn)
  elastic-testkit/      # simulator + fault injection (depends on core+rt)
  elastic/              # facade + prelude + examples + macro tests
```

**Zero SLHA dependencies, by construction**: no crate in `elastic/` imports
`scirust`, `slha-*` or `slhav2-vram`, and the standalone extraction test
(below) proves it by building the subtree outside the SLHAv2 workspace.

## Extraction procedure (exact commands)

```bash
# 1. Copy the subtree to a fresh location.
mkdir -p /tmp/elastic-standalone && cp -r elastic/* /tmp/elastic-standalone/

# 2. Build and test the standalone workspace (no SLHAv2 anywhere).
cd /tmp/elastic-standalone
cargo test --workspace

# 3. Prove the dependency graph is clean.
cargo tree --workspace | grep -iE 'scirust|slha|ccos|llama' \
  && echo "LEAK FOUND" || echo "CLEAN: no SLHA/CCOS dependency"

# 4. (Later) create the dedicated repository with git filter-repo:
#    git filter-repo --path elastic/ --path-rename elastic/:.
```

The workspace manifests use `version.workspace = true` etc. from the
incubator root `Cargo.toml`; when extracting, replace them with literal
values (the dedicated repo's own `[workspace.package]` can keep the same
structure, so even that is a one-line change).

## Rules that keep extraction trivial

1. **Never import SLHAv2** (or CCOS, llama.cpp, CUDA, FLAT-ATTENTION) from
   any `elastic/*` crate. The dependency arrow points UP into SLHAv2, never
   down.
2. **Keep elastic-core dependency-free** (`no_std + alloc`, `forbid
   (unsafe_code)`). Any new dependency must be justified and optional.
3. **Keep the facade as the only public surface**; application code should
   not need to name the four inner crates.
4. **Keep the macro crate's deps to proc-macro2/quote/syn.**
5. **No `publish = false` barriers**: the crates are unpublished today, but
   the manifests must remain publishable later without changes.

## Verification run (this mission)

The standalone extraction test is executed as part of the mission gates and
recorded in `docs/ELASTIC_MISSION_RESULTS.md`. Command:

```bash
rm -rf /tmp/elastic-standalone && mkdir -p /tmp/elastic-standalone
cp -r elastic/* /tmp/elastic-standalone/
cd /tmp/elastic-standalone && cargo test --workspace
```

## The user's dedicated repository

The user will create the dedicated Elastic language project after this
mission. The extraction procedure above is the documented handoff; nothing
in SLHAv2's `elastic/` requires the SLHAv2 tree to build or test.
