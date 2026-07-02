# Premiers pas avec SLHA v2

Ce guide vous prend par la main pour comprendre et utiliser SLHA v2, même si
vous n'avez jamais fait d'IA.

---

## 1. C'est quoi SLHA v2 ?

Imaginez une IA qui parle. Pour répondre, elle doit se souvenir de ce que vous
avez dit avant. Ce souvenir grandit à chaque mot, et finit par saturer la
mémoire de l'ordinateur.

**SLHA v2** compresse ce souvenir pour qu'il tienne dans un mouchoir de poche.
Résultat : l'IA tourne sur un PC normal, pas besoin d'une carte graphique à
2000 €.

---

## 2. Installation

### Prérequis

- Un ordinateur sous **Linux**, **macOS** ou **Windows** (avec WSL)
- **Rust** installé (le langage de programmation, pas le jeu vidéo)

Si vous n'avez pas Rust, installez-le en une commande :

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Option A : Installeur automatique (recommandé)

```bash
curl -sSL https://raw.githubusercontent.com/CHECKUPAUTO/SLHAv2/master/install.sh | bash
```

Ce script installe Rust (si besoin), clone le projet, le compile et lance les
tests. En 2 minutes c'est prêt.

### Option B : Installation manuelle

```bash
git clone https://github.com/CHECKUPAUTO/SLHAv2.git
cd SLHAv2
cargo build --release
cargo test
```

---

## 3. Premier contact

Lancez l'exemple de base :

```bash
cargo run --example basic_usage
```

Vous devriez voir :

```
Score: -8.000000
Tile is in HOT mode (full fidelity)
Dequantized latent[0..4]: [-4.0, -4.0, -4.0, -4.0]
```

**Ce qui vient de se passer :**
1. Un vecteur de 128 dimensions a été compressé en 64 octets
2. SLHA a calculé un « score d'attention » entre ce vecteur et une requête

---

## 4. Comprendre les concepts (sans maths)

### Le KV-cache (le souvenir de l'IA)

```
Personne : "Bonjour, comment ça va ?"
IA : "Je vais bien !"
Personne : "Tu te souviens de ce que j'ai dit ?"
                           ↑
              L'IA doit retrouver "Bonjour, comment ça va ?"
              dans son KV-cache pour répondre.
```

### Pourquoi c'est un problème

| Sans compression | Avec SLHA v2 |
|---|---|
| 1 token = ~2048 octets | 1 token = **128 octets** |
| Pour 100 000 tokens = **200 Mo** | Pour 100 000 tokens = **12 Mo** |
| Doit tenir en RAM/VRAM | Empreinte plus petite (reste plus longtemps résident) |
| Limité par la bande passante RAM | Moins d'octets/token à lire |

> **Ce sont des projections de compression, pas toutes mesurées.** Le ratio
> **mesuré** au niveau kernel est 128 o vs 256 o pour une clé bf16 = **2× moins
> d'octets/token** (§7.5) ; le débit dépend du CPU et de l'auto-vectorisation
> (~2,5× tokens/s sur un banc Xeon AVX2, ~1,3× sur CPU scalaire). La résidence
> **en cache** n'est qu'un effet indirect (compteurs `perf` indisponibles dans
> le banc, §6.1) : 12 Mo ne tiennent pas en L1/L2 (32 Ko–2 Mo), mais la
> empreinte réduite garde le working set résident plus longtemps. Le gain de
> bout en bout sur un LLM réel reste à mesurer.

### Les deux composants

SLHA v2 divise chaque souvenir en deux :

1. **La base** (80% de l'info, 64 octets) — une version compressée en 4 bits
   par dimension (comme une photo JPEG basse résolution)
2. **Le correctif** (20% de l'info, 32 octets) — des bits de correction qui
   rattrapent les erreurs de la compression

Le score final combine les deux : `score = base + λ × correctif`

### Le Soft-Paging

Quand la mémoire sature, SLHA peut **jeter le correctif** et ne garder que la
base (tuile WARM = 96 o contre 128 o en HOT, soit **−25 %** d'empreinte).
Mesuré : pager la moitié des tuiles HOT→WARM laisse la sortie d'attention à
**cos ≈ 0,9995** vs tout-HOT (§4) — une déviation de l'ordre de 0,05 %, pas 5 %.

---

## 5. Mesurer les performances

```bash
# Benchmark complet (fidélité, débit, HOT vs WARM)
cargo run --example measure --release

# Comparaison débit mémoire vs BF16
cargo run --example bench_vs_fp16 --release

# Micro-benchmarks (AVX2, AVX-512, scalaire)
cargo bench
```

`measure` affiche une table **HOT vs WARM** par énergie résiduelle `rho`, puis le
débit **scalaire / AVX2 / AVX-512**. Extrait représentatif (chiffres réels) :

```
   rho |   HOT Spear   top16 |  WARM Spear   top16
  0,05 |       0,987   0,875 |       0,987   0,875
  0,50 |       0,844   0,500 |       0,811   0,438

3) Débit : scalaire ~3 M/s · AVX2 ~34 M/s · AVX-512 ~41 M/s
```

---

## 6. Intégrer dans votre projet

### Si vous codez en Rust

```rust
// 1. Ajouter la dépendance
// [dependencies]
// scirust = { git = "https://github.com/CHECKUPAUTO/SLHAv2" }

use scirust::attention::slha_v2;

// 2. Compresser un vecteur
let mon_vecteur = [0.5f32; 128];
let (packed, scale) = slha_v2::quantize_latent(&mon_vecteur);

// 3. Calculer un score
let score = ma_tuile.compute_score(&ma_requete, &ma_signature);
```

### Si vous utilisez llama.cpp / Ollama

Voir le [guide d'intégration complet](INTEGRATION.md).

---

## 7. Aller plus loin

| Vous voulez... | Lisez... |
|---|---|
| Comprendre les maths | [`SLHAv2.md`](../SLHAv2.md) |
| Voir les résultats de mesures | [`FINDINGS.md`](../FINDINGS.md) |
| Intégrer dans un moteur LLM | [`INTEGRATION.md`](INTEGRATION.md) |
| L'API technique complète | [`api.md`](api.md) |
| Contribuer au code | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |

---

## 8. FAQ

**Q : Ça marche sur mon Raspberry Pi ?**
R : Le kernel a un chemin ARM NEON et compile sur AArch64 ; le kit
`examples/platform_report` détecte et rapporte les capacités du CPU. À ce jour
aucun banc n'a été exécuté sur Raspberry Pi — mesurez le vôtre
(`cargo run --example platform_report --release`).

**Q : Quel est l'impact sur la qualité des réponses ?**
R : La fidélité de la **sortie** d'attention est mesurée à cosinus 0,95-0,997 vs
l'attention complète — sur des données **synthétiques** (cf. `FINDINGS.md`,
§7.6). C'est un proxy de la perplexité accessible hors LLM réel ; aucune
évaluation humaine ni mesure sur vrai modèle n'a été faite.

**Q : Je peux l'utiliser avec mon propre modèle ?**
R : Oui, SLHA v2 est un composant que vous branchez dans votre pipeline
d'inférence. Il ne remplace pas le modèle, il optimise sa mémoire.

**Q : C'est gratuit ?**
R : Les usages non-commerciaux et personnels sont gratuits sous la licence
PolyForm Noncommercial 1.0.0 (voir `LICENSE.md`). Tout usage commercial
requiert une licence commerciale séparée, offerte pour les déploiements
**CCOS** — SLHAv2 et TurboQuant sont des modules compagnons de CCOS
(voir `LICENSING.md`).

---

*Prochaine étape : [Guide d'intégration](INTEGRATION.md) →*
