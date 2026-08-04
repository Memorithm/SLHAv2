# SLHA v2 — Faites tourner une IA locale sans carte graphique

[![CI](https://github.com/Memorithm/SLHAv2/actions/workflows/ci.yml/badge.svg)](https://github.com/Memorithm/SLHAv2/actions)
[![Rust](https://img.shields.io/badge/rust-2021+-blue.svg)](https://rust-lang.org)

---

Faire tourner une IA en contexte long chez soi exige normalement une carte
graphique hors de prix : à chaque mot généré, l'IA doit se souvenir de tout ce
qui précède, et ce « souvenir » — le **KV-cache** — grossit sans cesse jusqu'à
saturer la VRAM.

**SLHA v2** compresse ce KV-cache en tuiles de **128 octets** alignées
cache-line — l'équivalent d'une ligne de texte par token, au lieu de plusieurs
kilo-octets. Deux lignes de cache de 64 octets, conçues pour rester proches du
processeur (caches L1/L2/L3) plutôt que de dépendre d'un GPU.

> **Concrètement (projection) :** en compressant le KV-cache, un LLM qui
> nécessite ~8 Go de VRAM pourrait tenir sur un CPU avec ~4 Go de RAM. C'est
> l'objectif du projet — **à valider sur un modèle réel** : aucune mesure de bout
> en bout n'existe encore (voir les *Réserves d'honnêteté* plus bas).

---

## Relation avec le contrat mémoire CCOS d'OpenClaw

SLHA v2 est un **noyau de compression de KV-cache** ; son
`scirust/src/ccos.rs` (`ElasticKvCache`, arena KV Soft-Paging) est un artefact
**différent** du serveur mémoire CCOS — il ne partage que le nom « CCOS ». Le
serveur `slha-mcp` expose des outils `slha.*`
(audit / explain / compress / score / benchmark), **pas**
`ccos.recall` / `get` / `sync`. OpenClaw atteint le vrai serveur mémoire CCOS
via `mcporter` (serverName `ccos`) sur
[Memorithm/CCOS](https://github.com/Memorithm/CCOS). Ne pas renommer
`serverInfo.name` en `ccos` (cela casserait le routage mcporter).

## Comment ça marche (en 30 secondes)

Quand une IA génère du texte, elle doit se souvenir de tout ce qui a été dit
avant. Ce « souvenir » (le **KV-cache**) grossit à chaque mot et sature la
mémoire.

SLHA v2 compresse chaque souvenir en une **tuile de 128 octets** — l'équivalent
d'une ligne de texte — au lieu de plusieurs kilo-octets normalement.

| Sans SLHA v2 | Avec SLHA v2 |
|---|---|
| ~500 Mo pour 32k tokens¹ | ~4 Mo pour 32k tokens¹ |
| Obligé d'avoir un GPU | Fonctionne sur CPU |
| RAM saturée rapidement | Cache L1/L2/L3 utilisé intelligemment |

> ¹ *Projection* par tuile de 128 o/token (basée sur une clé non compressée
> ~15,6 ko/token). Le ratio **mesuré** au niveau kernel est 128 o vs 256 o pour
> une clé bf16 = **2× moins d'octets/token** (§7.5) ; le facteur de bout en bout
> sur un LLM réel reste à mesurer.

## Le projet en bref — ce que vous pouvez faire

SLHA v2 est un **workspace Cargo de 4 crates** (tous en v0.2.0), organisé autour d'un noyau de référence et de ponts vers l'extérieur. Concrètement, avec ce dépôt vous pouvez :

- **Compresser** chaque souvenir de KV-cache en une **tuile de 128 octets** sans padding (latent bas-rang 64 o + résidu 1-bit 32 o + métadonnées 32 o), et **scorer** une requête contre cette tuile sans la décompresser : produit scalaire sur le latent + popcount/Hamming sur le résidu (`compute_score`, eq. 2.3). Quatre codecs latents au même budget de 64 o : **INT4** (simple ou groupé MX), **NF4**, **mixte 8/4-bit**, et **TQ3** — portage du codec [TurboQuant](docs/TURBOQUANT.md) (3 bits + plan de correction 1 bit séparable).
- **Piloter le cache mémoire** avec **CCOS** (`ccos::ElasticKvCache`), un cache KV élastique « Soft-Paging » sur arène contiguë (états HOT 128 o / WARM 96 o, résidu masqué + λ=0 / COLD évincé) qui borne l'empreinte sous un budget en octets que vous fixez (`enforce_budget`), avec politiques de pagination (σ_E / ancienneté) et d'éviction (Causal par défaut, ou Importance H2O/StreamingLLM).
- **Filtrer la sécurité directement sur le latent compressé** : le module **`safety`** (`LatentSafetyGuard`) détecte injection et dérive **avant** décompression — déviation angulaire (cosinus), isolation orthogonale (classifieur linéaire optionnel), dérive glissante (fenêtre de 4). ~200 cycles/tuile, zéro allocation, safe Rust portable. Module **additif** : il n'altère ni la tuile 128 o ni les kernels SIMD.
- **Aligner et placer la mémoire** (module `numa`, *additif*) : `AlignedBuffer` (allocation alignée portable, zéro dépendance, disponible par défaut partout) ; en option la feature **`numa`** (Linux, `libc` en dépendance optionnelle) ajoute placement **NUMA** (`mmap`/`mbind` best-effort), épinglage de thread (`sched_setaffinity`) et introspection sysfs, avec repli gracieux ailleurs (`NumaError::Unavailable`).
- **Auditer** votre build avec le binaire **`slha-audit`** : layout de tuile (128 o, zéro padding, alignement), équivalence SIMD ≡ scalaire vérifiée à l'exécution, features CPU et niveaux de cache, fidélité vs attention complète, invariant de budget CCOS, déterminisme. Rapports Markdown ou JSON (`--json`/`--pretty`/`--out`), diff de régression (`--diff PRIOR.json`), sortie ≠ 0 en cas d'échec.
- **Brancher un agent** via le serveur **MCP** `slha-mcp` (stdio, JSON-RPC 2.0 délimité par lignes, **zéro dépendance externe** — réutilise `scirust::json`), qui expose 5 outils : `slha.audit`, `slha.explain`, `slha.compress`, `slha.score`, `slha.benchmark`.
- **Appeler le noyau depuis d'autres langages** grâce aux bindings **C** (`slha-c`, interface ABI C `cdylib`/`staticlib` + en-tête `slha.h`) et **Python** (`slha-python`, module natif via PyO3).

Le noyau **`scirust`** est à **zéro dépendance externe par défaut** (build offline), avec dispatch SIMD choisi **à l'exécution** (AVX-512 > AVX2 > scalaire sur x86_64, NEON sur aarch64 ; repli scalaire portable, aucun gating à la compilation). L'équivalence SIMD ≡ scalaire du score est garantie **à 1e-3 près** (FMA / accumulation réordonnée, pas bit-pour-bit) ; seul le popcount/Hamming est exact bit-à-bit. Le noyau embarque aussi les modules `safety`, `numa`, `incoherence` (RHT de Hadamard opt-in), `rope`, `residual`, `adapter`, `ccos`, `audit` et `json`, et fournit le binaire `slha-audit`. Les ratios SIMD sont **indicatifs** et dépendent de votre matériel — mesurez les vôtres avec `cargo run --example cycles --release`.

> **Licence :** double licence — **PolyForm Noncommercial 1.0.0** (usage non-commercial et personnel, gratuit) ; **licence commerciale** requise pour tout usage commercial (voir `LICENSING.md`). SLHAv2 et [TurboQuant](https://github.com/CHECKUPAUTO/TurboQuant) sont des modules compagnons de **CCOS** ; la licence commerciale est offerte exclusivement pour les déploiements CCOS (l'usage non-commercial reste régi par PolyForm NC, sans restriction d'environnement).

> Le dépôt est un **workspace Cargo** : toutes les commandes ci-dessous se
> lancent depuis la racine.

---

## Installation (30 secondes)

```bash
git clone https://github.com/Memorithm/SLHAv2.git
cd SLHAv2
git rev-parse HEAD
less install.sh
./install.sh

# Installation manuelle
cargo build --locked --release -p scirust -p slha-mcp -p slha-c
cargo test --locked -p scirust -p slha-mcp -p slha-c
```

**Prérequis :** Git et Rust doivent être installés au préalable.
L’installeur local ne télécharge aucun script et ne supprime aucun répertoire.

### Installation augmentée — auto-détection et branchement agent

Pour que l'installation **détecte elle-même** votre environnement (agent IA,
moteur LLM, GPU, toolchains) et **branche automatiquement le serveur MCP**
auprès de votre agent :

```bash
./install.sh --auto --yes        # détecte + construit + teste + branche l'agent
./install.sh --detect            # juste le rapport d'environnement (rien d'autre)
```

`--auto` détecte l'agent (Claude Code / Claude Desktop / Cursor), le moteur
LLM local (Ollama / vLLM / llama.cpp) et ses modèles, la GPU NVIDIA + version
CUDA, le CPU/RAM, puis s'enregistre auprès de l'agent détecté — l'agent dispose
alors des outils `slha.*` sans aucune commande manuelle. L'enregistrement est
best-effort : si aucun agent n'est utilisable, l'installation réussit quand
même et explique quoi faire. Détails et drapeaux (`--with-cuda`, `--skip-*`,
`--register-mcp`, …) : [`docs/AUTO_INSTALL.md`](docs/AUTO_INSTALL.md).

---

## Premier essai

```bash
# Voir ce que ça donne sur votre machine
cargo run --example basic_usage

# Mesure de performance complète
cargo run --example measure --release

# Proxy de décodage bout-en-bout (⚠️ PAS un vrai LLM) : tok/s SLHA vs réf f32
cargo run --example benchmark_decode --release
```

Sortie de `basic_usage` :
```
Score: -8.000000
Tile is in HOT mode (full fidelity)
Dequantized latent[0..4]: [-4.0, -4.0, -4.0, -4.0]
```

---

## Auditer le système (`slha-audit`)

Un outil dédié vérifie que tout est sain et **génère un rapport** (Markdown ou
JSON) — pratique en CI, pour un agent, ou comme trace d'audit :

```bash
cargo run --bin slha-audit                              # rapport Markdown lisible
cargo run --bin slha-audit -- --json --out audit.json   # rapport machine (JSON)
cargo run --bin slha-audit -- --diff audit.json         # diff vs un rapport antérieur (régression)
```

Il contrôle, **à l'exécution** : le layout de tuile (128 o, zéro padding,
alignement), l'**équivalence SIMD ≡ scalaire** (le chemin AVX-512/NEON rend le
même score que la référence), les features CPU + niveaux de cache, la **fidélité
de sortie** vs attention complète, l'**invariant de budget CCOS**, et le
**déterminisme**. Code de sortie ≠ 0 si un contrôle échoue.

Pour éprouver tout le produit d'un coup (gate qualité + tous les exemples +
rapport horodaté) : `./scripts/stress_test.sh`.

---

## Connecter un agent / LLM (MCP)

Un serveur **MCP** (`slha-mcp`, **zéro dépendance**) expose le noyau et l'audit
SLHA comme **outils appelables par un agent** (Claude Code / Desktop, ou tout
client MCP) :

```bash
cargo build --release -p slha-mcp
claude mcp add slha -- "$(pwd)/target/release/slha-mcp"
```

L'agent dispose alors de 5 outils : `slha.audit`, `slha.explain`,
`slha.compress`, `slha.score`, `slha.benchmark`. Config Claude Desktop, schéma
des outils et exemple de session : [`docs/MCP.md`](docs/MCP.md).

---

## Intégrer SLHA v2 dans mon projet

### Projet Rust

Ajoutez à votre `Cargo.toml` :

```toml
[dependencies]
scirust = { git = "https://github.com/Memorithm/SLHAv2" }
```

Puis dans votre code :

```rust
use scirust::attention::slha_v2;

// Compresser un vecteur de clé (128 dims -> 64 octets INT4)
let mon_vecteur = [0.5f32; 128];
let (packed, scale) = slha_v2::quantize_latent(&mon_vecteur);

// Créer une tuile compressée (128 octets)
let tuile = slha_v2::SciRustSlhaTile {
    latent_kv: packed,
    residual_bitmap: [0u64; 4],
    scale,
    dynamic_lambda: 0.5,
    residual_sigma: 0.0,
    token_id: 0,
    position: 0,
    head_id: 0,
    flags: slha_v2::FLAG_HOT,
    group_scales: [255u8; 8],
};

// Calculer le score d'attention (dispatch SIMD automatique)
let q_coarse = [0.0f32; 128];
let q_sign = [0u64; 4];
let score = tuile.compute_score(&q_coarse, &q_sign);
```

### Intégration avec llama.cpp / Ollama / vLLM

Voir le [guide d'intégration](docs/INTEGRATION.md) — **esquisse de conception**
(pseudo-code), pas une intégration livrée.

---

## Documentation

| Document | Pour qui | Contenu |
|---|---|---|
| [**Premiers pas**](docs/GETTING_STARTED.md) | Débutants | Installation, premier essai, concepts |
| [**Connexion MCP**](docs/MCP.md) | Agents / LLM | Brancher un agent sur les outils SLHA (audit, score, benchmark) |
| [**Guide d'intégration**](docs/INTEGRATION.md) | Développeurs | Esquisse pour llama.cpp, Ollama, vLLM |
| [**Spécification**](SLHAv2.md) | Chercheurs | Maths complètes + résultats §7 |
| [**Résultats**](FINDINGS.md) | Curieux | Ce qu'on a mesuré, ce qui marche, ce qui reste |
| [**API Reference**](docs/api.md) | Développeurs | Documentation technique (API réelle) |
| [**Détails du crate**](scirust/README.md) | Développeurs | Organisation de `scirust/` |

---

## État du projet

- ✅ **Mécanisme validé** : suite workspace complète couvrant le noyau, l’auto-audit, CCOS, l’ABI C, le serveur MCP et le binding Python ; tests unitaires, intégration, propriétés (property-based), doctests et parité SIMD ≡ scalaire par ISA ; Clippy `-D warnings` et CI
- ✅ **Performance** : x86 **AVX2 ~×11,5 / AVX-512 ~×14,1** (banc Xeon partagé) ; ARM **NEON ~×5,7** (Jetson Thor AGX 128) — vs scalaire. _Ratios **indicatifs**, dépendants du CPU et de l'auto-vectorisation ; mesurez les vôtres : `cargo run --example cycles --release`._
- ✅ **Multi-plateforme** : x86_64 (AVX2/AVX-512/VPOPCNTDQ) + ARM AArch64 (NEON, **mesuré sur Jetson Thor** ; `sve2` détecté) — kit `examples/platform_report`
- ✅ **Fidélité** : cosinus 0,95–0,997 vs attention complète (sortie `softmax·V`)
- ✅ **Soft-Paging** : cache KV élastique (`ccos::ElasticKvCache`) — pager la moitié des tuiles HOT→WARM laisse la sortie à **cos 0,9995** (`examples/ccos_softpaging`, §4)
- ✅ **Auto-audit + accès agent** : outil `slha-audit` (rapports JSON/Markdown) et serveur **MCP** `slha-mcp` (5 outils, zéro dépendance) — [`docs/MCP.md`](docs/MCP.md)
- 🟡 **Intégration LLM réel** : *esquisse* — guide de conception + croquis pour llama.cpp/vLLM disponibles ([`docs/INTEGRATION.md`](docs/INTEGRATION.md)), **non intégrée** dans un moteur d'inférence ; perplexité non mesurée.

> Réserves d'honnêteté (projections synthétiques, `perf`/perplexité hors banc) :
> voir [`FINDINGS.md`](FINDINGS.md) et `SLHAv2.md` §6–7.

---

## Contribuer

Les contributions sont les bienvenues — voir [`CONTRIBUTING.md`](CONTRIBUTING.md)
et les [issues](https://github.com/Memorithm/SLHAv2/issues).

```bash
git clone https://github.com/Memorithm/SLHAv2.git
cd SLHAv2
cargo test --workspace --all-features   # suite complète, tous les membres
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

---

## Licence

Dual-licensed: [PolyForm Noncommercial 1.0.0](LICENSE.md) for noncommercial and personal
use; commercial license required for any commercial use.
See [LICENSING.md](LICENSING.md).

— [Forge CHECKUPAUTO](https://github.com/CHECKUPAUTO)
