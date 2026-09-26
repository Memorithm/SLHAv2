# SLHA v2 - research paper

LaTeX source for the research paper:

> **SLHA v2: Cache-Aware Hybrid Attention with Low-Rank Latents and 1-Bit
> Sign-LSH Residuals for Memory-Bandwidth-Bound LLM Inference**
> *Tarek Zekriti (ZEKRITI TAREK)*

## September 2026 revision

The paper is maintained as an evidence ledger rather than silently rewriting
earlier claims:

- **blue** text is added or corrected in the September 2026 review;
- **red** text is superseded/deleted material retained visibly for audit.

The revision preserves the original seeded synthetic mechanism studies while
adding the repository's later evidence: real GPT-2 activation codec
requalification, Qwen2.5-1.5B-Instruct/WikiText-2 perplexity gates, the
TinyStories physical external-K CCOS experiment, rank-transplant diagnostics,
durable EventLog-backed COLD eviction, the fixed-slot physical cache,
ElasticXxx source-bound capacity/precision contracts, and the RPL-0 replayable
state contract.

The paper explicitly records negative results. In particular, the tested
direct compressed-score replacement configurations currently fail real-model
quality gates, and the validated physical llama.cpp experiment is **K-only**
(V remains in the ordinary engine cache). Cache-owned residency counters are
not presented as total process/GPU memory savings.

## September 26 adaptive-KV extension

The revision now also records the evidence-driven pivot from a single fixed
compressed-score design toward an **importance-preserving adaptive KV**
programme. The new material is blue and is explicitly prospective: it does not
turn the current real-model NO-GO into a positive result.

The extension adds:

- current 2025 comparison classes (low-rank projection, adaptive/layer-wise
  mixed precision, vector quantization, coarse+fine sparse KV, query-agnostic
  importance);
- ranking/top-set preservation and retained softmax mass as separate
  mechanistic objectives;
- KVLab as the upstream falsification bench;
- ElasticXxx as the generic representation-transition controller;
- FLAT-ATTENTION and NNIS as the portable/native physical qualification paths;
- explicit rules preventing logical candidate density or bit width from being
  presented as physical speed/memory evidence.

The canonical implementation roadmap is `../ROADMAP.md` and the evidence
handoff contract is `../docs/KVLAB_FEEDBACK_CONTRACT.md`.

## Build

Self-contained -- only standard LaTeX packages, no external `.bib`, no custom
`.sty`. Two passes resolve cross-references and the bibliography.

```sh
pdflatex -interaction=nonstopmode -halt-on-error slhav2.tex
pdflatex -interaction=nonstopmode -halt-on-error slhav2.tex
```

or:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error slhav2.tex
```

The repository CI rebuilds and render-checks the committed PDF when the paper
source changes. Research branches that modify the paper are included in this
rebuild gate so the committed PDF cannot silently drift from the LaTeX source.

## arXiv submission

For a publication-clean version, remove the revision wrappers (`\added`,
`\deleted`, `addedblock`, `deletedblock`) after the review is accepted. The
bibliography remains embedded via `thebibliography`. Suggested primary category:
`cs.LG`; cross-list `cs.AR`, `cs.PF`.
