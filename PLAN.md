# Plan d'action v2 — Valider (ou réfuter) SLHA v2 sur un vrai LLM

> **Étoile polaire :** prouver la thèse — « la compression latent + résidu 1-bit
> préserve la qualité du modèle » — sur un modèle réel, **le moins cher
> possible d'abord**. Tout le reste (portabilité, packaging) est secondaire tant
> que ce GO/NO-GO n'est pas chiffré.
>
> **Ce qui change vs le plan v1 :** (1) une **Phase 0 de validation offline** est
> placée avant toute intégration lourde ; (2) l'**entraînement des projections**
> passe *avant* la mesure de perplexité (sans lui, brancher SLHA dans un LLM
> donne une perplexité catastrophique et une conclusion fausse) ; (3) le plan est
> **réconcilié avec le master** (≈ 1/3 des items v1 sont déjà codés) ; (4) les
> items qui heurtent la **licence** ou l'**invariant 128 o** sont corrigés.

---

## 0. Déjà fait sur `master` (ne pas re-planifier — à *valider*, pas à écrire)

| Domaine | État |
|---|---|
| Noyau tuile 128 o, zéro padding | ✅ audité/testé (`slha-audit`) |
| SIMD AVX2 / AVX-512 / NEON + repli scalaire, ≡ testée | ✅ |
| Codecs INT4 (MX groupé) / NF4 | ✅ |
| CCOS Soft-Paging HOT/WARM/COLD + éviction informée (A5) | ✅ `ccos`, `informed_eviction` |
| Projection apprise (PCA + SGD) | ✅ `learned` |
| Pré-RoPE (A1), Hadamard/incohérence (A2), résidu multi-bit (A4) | ✅ `rope`, `incoherence`, `residual` + exemples |
| Proxies offline : fidélité de sortie, décodage bout-en-bout | ✅ `attention_fidelity`, `benchmark_decode` |
| NUMA / alloc alignée, filtre de sécurité latent | ✅ `numa`, `safety` |
| Outils : `slha-audit` (+ `--perf`), serveur MCP | ✅ |
| Bindings C (`slha-c`) et Python (`slha-python`) | ✅ |

**Conséquence :** ce plan concerne surtout la **validation sur activations
réelles** et quelques briques réellement neuves — pas la ré-implémentation de
l'existant.

---

## 1. Critère de succès — à figer AVANT d'écrire une ligne d'intégration

Sans seuil, on ne peut pas conclure. À fixer d'abord :

- **Fidélité :** Δ-perplexité relative vs FP16 ≤ **cible** (ex. ≤ 1–2 % en HOT sur
  le contexte cible), documentée pour HOT / WARM / COLD.
- **Mémoire :** empreinte KV vs baseline (viser le facteur mesuré, pas projeté).
- **Débit :** tokens/s de bout en bout à contexte long, vs baseline FP16.

Livrable : [`docs/SUCCESS_CRITERIA.md`](docs/SUCCESS_CRITERIA.md) — une page,
chiffrée, **seuils pré-enregistrés** (fait ✅).

---

## Phase 0 — Validation OFFLINE (le GO/NO-GO, peu coûteux) · **P0**

But : estimer l'impact qualité **sans toucher à llama.cpp**, en quelques jours.

| Étape | Description | Durée |
|---|---|---|
| **0.1** | Choisir un petit modèle ouvert (GPT-2 small / Pythia-160M) + WikiText-2 | 0,5j |
| **0.2** | Script Python (`transformers`) : dumper Q/K/V d'une (puis plusieurs) couche(s) → fichiers | 1–2j |
| **0.3** | Harnais Rust `examples/offline_validation.rs` : charge les K réels, **apprend la projection** (`learned::`), encode en tuiles, calcule score SLHA + sortie softmax·V, compare à l'attention FP32 réelle (réutilise `attention_fidelity` : cosinus de sortie, rel-L2, top-k) | 3–5j |
| **0.4** | Proxy de perplexité : rejouer la couche et mesurer ΔNLL / KL sur un échantillon | 2–3j |
| **0.5** | **Décision chiffrée GO/NO-GO** + balayage `(d_c, d_s, λ, RHT on/off, résidu 1/2/4-bit)` pour trouver le régime viable | 2–3j |

**Sortie : rapport chiffré.** Si NO-GO → on ajuste l'algo **ici**, pas après
3 semaines d'intégration. C'est l'expérience à plus haut ROI du plan.

> **Statut (juillet 2026) : exécutée sur GPT-2 couche 6 — et le plan a marché
> exactement comme prévu.** NO-GO initial (cos 0,834), diagnostic offline en
> heures (pas en semaines d'intégration), deux correctifs mergés — projection
> jointe (`fit_joint`, §1.3) et codec latent mixte 8/4-bit
> (`LatentCodec::Mixed`, invariant 128 o intact) — puis re-mesure held-out :
> **cos 0,966 / KL 0,12**, à 99,4 % du plafond du sous-espace 128 (0,971).
> Rapport complet : `FINDINGS.md` §5 ; critère : `docs/SUCCESS_CRITERIA.md` §3.

---

## Phase 1 — Projections apprises sur activations réelles (**prérequis** de la perplexité) · **P0**

La perplexité réelle n'a de sens qu'avec une projection *apprise* : les
projections aléatoires (JL) actuelles donneraient une perplexité désastreuse.

| Étape | Description | Composants | Durée |
|---|---|---|---|
| **1.1** | `train_projection` → module autonome + **format de poids** `(P, Z, config)` | `learned.rs`, nouveau `weights.rs` | 3–4j |
| **1.2** | Exemple `train_on_real_activations.rs` (réutilise les dumps de la Phase 0) | `scirust/examples/` | 3–5j |
| **1.3** | Objectif d'entraînement = reconstruction **ET** proxy perplexité (pas seulement le score) | `learned.rs` | 5j |
| **1.4** | Calibrer λ + capter σ_E sur activations réelles (recherche de ligne) | — | 2–3j |

Durée : ~2–4 semaines.

---

## Phase 2 — Intégration LLM réelle : perplexité + débit de bout en bout · **P1** (seulement après Phase 0 = GO)

| Étape | Description | Durée |
|---|---|---|
| **2.1** | Fork llama.cpp + sous-module `extern/SLHAv2` + backend d'encodage via `slha-c` (existe) | 2–3j |
| **2.2** | Remplacer le stockage KV par des `SlhaTile` | 1 sem |
| **2.3** | Remplacer K·Q par le score SLHA dans la boucle d'attention — ⚠️ **le point dur** : attention **ggml fusionnée** (ops custom, RoPE, masking, batching) | **2–4 sem, risque élevé** |
| **2.4** | Flag CLI `--kv-cache slha` | 1j |
| **2.5** | Mesurer perplexité (WikiText-2/PTB/PG-19), débit 1K→128K, cache (`slha-audit --perf` existe) | 1 sem |

> **vLLM/CUDA (v1 §1.2) : retiré du chemin critique.** Écrire un kernel GPU pour
> une technique dont l'argument est « tourner **sans GPU** » dilue le message
> (PagedAttention gère déjà le KV GPU). Déplacé en **annexe** « si contexte
> ultra-long côté GPU ».

---

## Phase 3 — Densifier l'avantage mémoire (après la preuve) · **P2**

| Étape | Description | Durée |
|---|---|---|
| **3.1** | Compression des **V** dans une **tuile V séparée** — ⚠️ **ne pas gonfler la tuile K de 128 o** (invariant cache-line audité). Mesurer la fidélité de sortie | 5–10j |
| **3.2** | Résidu multi-bit : le module **existe (A4)** → mesurer **1 vs 2 vs 4 bit** sur modèle réel | 3–5j |
| **3.3** | Sparsification / attention-sinks / window : N derniers tokens HOT, reste WARM/COLD (s'appuie sur `informed_eviction`/CCOS) | 5j |

---

## Phase 4 — Portabilité & preuve perf (avance en parallèle) · **P1–P2**

| Étape | Description | Durée |
|---|---|---|
| **4.1** | **Valider les compteurs cache** (`perf_event_open` direct) sur Xeon + ARM → **prouve la thèse du mur de bande passante** (réserve d'honnêteté §6.1) | 2–3j |
| **4.2** | SVE2 : le **diagnostic est déjà fait** (intrinsèques nightly-only, `asm!` invérifiable sans appareil). Action réelle = **valider un `asm!` sur un vrai Jetson Thor / Graviton 4**, sinon attendre la stabilisation des intrinsèques | 5j (+ appareil) |
| **4.3** | CI multi-arch (runner ARM self-hosted) + `slha-audit --perf` en CI (l'outil existe) | 2–3j |

---

## Phase 5 — Packaging & diffusion · **P2** (⚠️ licence)

| Étape | Description | Durée |
|---|---|---|
| **5.1** | Paquet Debian `.deb` (lib + en-tête C + binaires) | 2j |
| **5.2** | Image Docker `checkupauto/slha-v2` | 1j |
| **5.3** | Extension MCP : outils `slha.train_projection` / `slha.page_out` / `slha.evict` (le serveur MCP existe) | 3j |

> ⚠️ **Publication publique crates.io / PyPI (v1 §5.4–5.5) : incompatible avec la
> licence `PolyForm-Noncommercial-1.0.0` (produit commercial dual).** Ne pas
> diffuser le source publiquement. Alternatives : registry privé, distribution
> **binaire** sous licence commerciale, ou une éventuelle lib « community »
> séparée si la décision produit le justifie.

---

## Priorités (revues)

| Priorité | Tâche | Pourquoi |
|---|---|---|
| **P0** | Phase 0 — validation offline | ROI max, coût min ; décide GO/NO-GO **avant** d'investir |
| **P0** | Phase 1 — projections apprises | prérequis d'une perplexité crédible (levier #1) |
| **P1** | Phase 2 — intégration llama.cpp | la **vraie** mesure ; chère/risquée → seulement après GO |
| **P1** | 4.1 — compteurs cache | prouve la thèse bande passante |
| **P2** | 3.1 V (tuile séparée), 3.2 multi-bit, 4.2 SVE2, 4.3 CI | densifient / portent une fois la preuve acquise |
| **P3** | vLLM GPU, bindings WASM/C++/Java | hors chemin critique |

---

## Garde-fous (principes du projet)

- **Invariant 128 o** intouchable : la compression V vit dans une **tuile
  séparée**, jamais dans la tuile K.
- **Rien de public sous licence non-commerciale.**
- **Mesurer, pas affirmer** : tout chiffre vient d'un test/exemple reproductible
  (graine fixe), jamais d'une estimation. Le critère de succès est figé **avant**
  l'intégration.
- **Le bon ordre** : offline pas cher → projection apprise → perplexité réelle →
  puis densification/portabilité/packaging.
