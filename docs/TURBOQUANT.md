# TQ3 — le codec TurboQuant porté dans la tuile SLHA v2

> Étude de faisabilité **réalisée et mesurée** : le codec KV-cache de
> [TurboQuant](https://github.com/CHECKUPAUTO/TurboQuant) (QJL : grille 3 bits
> + correction de signe 1 bit) tient **exactement** dans le budget latent de
> 64 octets de la tuile SLHA v2, sans toucher à l'invariant 128 o. Implémenté
> sous `LatentCodec::Tq3` / `FLAG_TQ3`.

## 1. Le constat de compatibilité

TurboQuant et SLHA v2 compressent le KV-cache par deux moyens orthogonaux :

| | TurboQuant | SLHA v2 |
|---|---|---|
| Réduction de **dimension** | non (rotation seulement) | oui (projection bas-rang D→128) |
| Réduction de **précision** | 3 bits + correction 1 bit | INT4 / NF4 / mixte |
| Anti-outliers | rotation orthogonale (PolarQuant) | blanchiment (`1/s_k`) + RHT opt-in |
| Correction résiduelle | 1 bit **par dimension** (quantification) | 256 bits **par tuile** (troncature de rang) |

Le budget tombe juste : `128 dims × 3 bits = 48 o` de codes + `128 × 1 bit
= 16 o` de signes de correction = **64 octets pile**, le plan latent de la
tuile. Résidu (32 o) et métadonnées (32 o) inchangés — la tuile reste 128 o,
zéro padding (vérifié par le test de layout et `slha-audit`).

## 2. Ce qui est porté, et ce qui ne l'est pas (et pourquoi)

**Porté** — la moitié « QJL » de TurboQuant :
- grille symétrique 8 niveaux `{±0,5 ; ±1,5 ; ±2,5 ; ±3,5}` (pas de niveau
  zéro), codes 3 bits en flux little-endian (`TQ3_CODE_BYTES` = 48 o) ;
- correction 1 bit par dimension : le bit déplace le niveau décodé de
  ±`TQ3_CORRECTION` (= 0,25 pas, l'amplitude optimale pour un résidu
  uniforme) — plan séparé de 16 o (`TQ3_CORR_BYTES`) ;
- échelles par groupe de 16 dims (`gs[8]`), mêmes conventions que les codecs
  INT4 groupé / NF4 / mixte.

**Non porté** — la moitié « PolarQuant » (rotation pré-quantification) :
`learned::LearnedModel` **blanchit déjà** le latent (division par `s_k`),
ce qui égalise la dynamique par dimension — le travail que la rotation de
TurboQuant fait sur un vecteur brut. Pour le résidu, la RHT du module
`incoherence` (axe A2) joue déjà ce rôle, en opt-in. Ajouter une rotation du
latent par-dessus la PCA détruirait en outre l'ordre par variance décroissante
qu'exploitent les échelles par groupe et le codec mixte.

## 3. Mesures (synthétique, graines fixes — reproductibles)

`cargo run --release --example offline_validation -- --codec tq3` :

| régime | état | cos↑ | KL↓ | verdict |
|---|---|---|---|---|
| decay=0.99 | HOT | 0,9979 | 0,0026 | GO ✅ |
| decay=0.95 | HOT | 0,9998 | 0,0003 | GO ✅ |
| decay=0.90 | HOT | 0,9999 | 0,0002 | GO ✅ |
| decay=0.80 | HOT | 0,9999 | 0,0001 | GO ✅ |

Baseline INT4 groupé sur les mêmes régimes : cos 0,9981 / 0,9998 / 0,9999 /
0,9999 — **TQ3 est au niveau des codecs 4 bits existants** (attendu : 3+1 bits
= 4 bits d'information par dimension, même erreur pire-cas de 0,25 pas).

`cargo run --release --example measure_learned` (Spearman end-to-end) :
INT4 groupé HOT 0,884 / WARM 0,609 ; **TQ3 HOT 0,881 / WARM 0,598**.

**Trade-off honnête** : la grille TurboQuant n'a pas de niveau zéro — toute
valeur proche de 0 paie ≥ 0,25 pas. Sur latent gaussien, MSE ≈ 1,3–1,6× celle
d'INT4 groupé (testé, garde-fou ≤ 2×). En end-to-end la différence est dans le
bruit (cf. ci-dessus), cohérent avec le finding « la quantification n'est pas
le goulot » sur données synthétiques.

**Ce que TQ3 apporte qu'aucun codec nibble ne peut offrir** : ses deux plans
sont **séparables**. Lâcher les 16 o de correction dégrade gracieusement vers
une tuile 3 bits pure — un barreau de pagination CCOS *plus fin* que HOT→WARM,
**implémenté** : `FLAG_TQ3_NOCORR` + `ElasticKvCache::drop_correction`,
échelle HOT 128 → HOT¬corr 112 → WARM 96 → WARM¬corr 80 → COLD 0 (ordre de
dureté strictement croissant, déterminisme testé).

## 3bis. Activations réelles (GPT-2 c6) — mesuré, NO-GO

Le §3 ci-dessus est **synthétique**. Rejoué le 2026-07-02 sur le protocole
« activations réelles » de [`FINDINGS.md`](../FINDINGS.md) §5 : GPT-2 small,
couche 6 (hook `c_attn`, d=768 pleine largeur), corpus train/test **disjoints**
de 1024 tokens chacun tirés de WikiText-2-raw-v1 (split *train* → `train.txt`,
split *test* → `test.txt`), projection entraînée sur le train et scorée
**tenue à l'écart** sur le test (`--weights`). Trois codecs, mêmes dumps,
lignes HOT (WARM ≈ HOT sur ce dump, écarts < 0,001) :

| protocole (HOT) | codec | cos↑ | relL2↓ | KL↓ | verdict |
|---|---|---|---|---|---|
| self-fit (optimiste) | INT4 groupé | 0,8056 | 0,6395 | 1,0431 | NO-GO ❌ |
| self-fit (optimiste) | mixte 8/4 | 0,9859 | 0,1531 | 0,0492 | NO-GO ❌ (KL seul) |
| self-fit (optimiste) | **TQ3** | 0,8347 | 0,6430 | 0,9053 | NO-GO ❌ |
| held-out, PCA-clés | INT4 groupé | 0,7582 | 0,6672 | 1,1616 | NO-GO ❌ |
| held-out, PCA-clés | mixte 8/4 | 0,9701 | 0,2102 | 0,1053 | NO-GO ❌ |
| held-out, PCA-clés | **TQ3** | 0,7831 | 0,6542 | 1,0768 | NO-GO ❌ |
| held-out, **JOINTE** (protocole §5) | INT4 groupé | 0,8835 | 0,4386 | 0,5804 | NO-GO ❌ |
| held-out, **JOINTE** (protocole §5) | mixte 8/4 | **0,9846** | 0,1492 | 0,0553 | NO-GO ❌ (KL seul) |
| held-out, **JOINTE** (protocole §5) | **TQ3** | **0,7908** | 0,6078 | 1,0956 | NO-GO ❌ |

Seuils GO (HOT) : cos ≥ 0,98 **et** KL ≤ 0,03 (`GO_COSINE`, `GO_KL`).

**Lecture honnête.** Le « TQ3 ≈ INT4 groupé » du §3 ne survit **pas** aux
activations réelles : sur le spectre raide de GPT-2 (40 % de l'énergie dans une
direction, 56× de dynamique dans un groupe de scaling — FINDINGS §5), la grille
uniforme 8 niveaux sans zéro s'effondre comme l'INT4 uniforme — cos 0,78–0,83,
KL ≈ 0,9–1,1, très loin des seuils GO et du codec mixte (0,9846 / 0,0553),
construit précisément pour cette dynamique. Fait notable : la projection jointe
relève l'INT4 groupé (0,758 → 0,884) mais laisse TQ3 quasi inchangé
(0,783 → 0,791) — le goulot est la **grille**, pas le sous-espace ; ni la
correction 1 bit (±0,25 pas) ni le blanchiment `1/s_k` ne couvrent la dynamique
intra-groupe. Verdict : TQ3 est un **NO-GO mesuré comme codec HOT sur
activations réelles** ; le codec mixte reste le choix pour les spectres réels,
TQ3 garde son intérêt propre (plans séparables → rung de pagination, §3) sur
les distributions plates où il est au niveau des codecs 4 bits.

Reproduction (mêmes chiffres, graines fixes dans les exemples — 0x5107 pour
l'entraînement, 0x5C0FF pour le self-fit ; le dump lui-même est déterministe,
pas d'échantillonnage ; torch 2.12.1 CPU, transformers 5.12.1) :

```bash
# corpus : WikiText-2-raw-v1, split train → train.txt, split test → test.txt (disjoints)
python scripts/dump_activations.py --model gpt2 --layer 6 --out DUMPS/train --file train.txt --max-tokens 1024
python scripts/dump_activations.py --model gpt2 --layer 6 --out DUMPS/test  --file test.txt  --max-tokens 1024
cargo run --release --example train_on_real_activations -- --dump DUMPS/train --joint --out p_joint.slhw
for c in grouped mixed tq3; do
  cargo run --release --example offline_validation -- --dump DUMPS/test --weights p_joint.slhw --codec $c
done
```

Les chiffres absolus diffèrent légèrement du tableau de FINDINGS §5 (mixte
joint 0,9846 ici vs 0,966 là-bas ; INT4 PCA-clés 0,758 vs 0,834) : corpus
différent (WikiText-2 ici, corpus non archivé pour §5). La chaîne qualitative
est identique — INT4 uniforme NO-GO, mixte la relève, la jointe aide l'INT4 —
et la conclusion TQ3 ne dépend pas du corpus.

## 3ter. La synthèse MIX3 — la réponse au NO-GO, mesurée

Le diagnostic du §3bis (« le goulot est la grille, pas le sous-espace »)
appelait sa synthèse : **`LatentCodec::Mix3`** = la tête 8-bit du codec mixte
(8 dims où est l'énergie) + un corps TQ3 (112 dims à 3 bits, plan de
correction 1 bit **séparable** de 14 o), queue lâchée — 8 + 42 + 14 =
**64 octets pile**. Rejoué sur le même protocole GPT-2 (held-out, jointe) :

| codec (HOT, held-out jointe) | cos↑ | relL2↓ | KL↓ |
|---|---|---|---|
| mixte 8/4 | 0,9846 | 0,1492 | 0,0553 |
| **MIX3** | **0,9835** | 0,1552 | 0,0599 |
| TQ3 | 0,7908 | 0,6078 | 1,0956 |

MIX3 est **au niveau du mixte sur activations réelles** (écart 0,001 de
cosinus — le coût de la grille sans zéro sur le corps) tout en offrant ce
que le mixte ne peut pas : le **barreau de pagination CCOS** (échelle
HOT 128 → HOT¬corr 114 → WARM 96 → WARM¬corr 82 → COLD 0, via le même
`FLAG_TQ3_NOCORR`/`drop_correction`, comptabilité 14 o testée). Synthétique :
GO sur tous les régimes (cos 0,9976–0,9999). Le KL strict (≤ 0,03) reste
au-dessus du seuil pour *tous* les codecs sur ce dump, mixte compris — c'est
la frontière projection/protocole, pas le codec.

## 3quater. La frontière projection — SGD score-aware : levier négatif, mesuré

Le §3ter établit que, sur activations réelles, le goulot n'est plus le codec
(mixte ≈ MIX3) mais la **projection** : le KL du meilleur codec (0,055) reste
au-dessus du seuil (≤ 0,03) alors que le cosinus (0,984) le passe largement.
Levier testé : **raffiner la projection jointe (PCA) par SGD score-aware**
(`train_projection`, objectif = erreur de *score*, plan §1.3), sur le même dump
GPT-2 held-out. Verdict : **négatif et instable.**

| projection (held-out jointe, HOT) | codec | cos↑ | relL2↓ | KL↓ |
|---|---|---|---|---|
| PCA jointe (référence §3ter) | mixte | **0,9846** | 0,1492 | **0,0553** |
| PCA jointe | MIX3 | 0,9835 | 0,1552 | 0,0599 |
| + SGD (50 ép., lr 1e-6, *convergé*) | mixte | 0,9210 | 0,3830 | 0,2910 |
| + SGD (50 ép., lr 1e-6, *convergé*) | MIX3 | 0,9219 | 0,3863 | 0,2890 |
| + SGD (300 ép. / lr ≥ 1e-5) | tous | *NaN* (diverge) | — | — |

**Deux échecs distincts.** (1) **Instabilité numérique** : le pas d'apprentissage
réglé sur le synthétique (2e-3) explose immédiatement en NaN sur les magnitudes
réelles de GPT-2 ; il faut descendre à **lr 1e-6** pour converger, et même là le
SGD finit par diverger au-delà de ~50 époques. (2) **Sur-apprentissage** : là où
il converge, le SGD réduit son objectif (erreur de score sur le *train*) mais
**dégrade la fidélité de sortie held-out** — cos 0,985 → 0,921, KL 0,055 → 0,291.
Il s'éloigne de l'optimum PCA qui, lui, généralise.

**Conclusion honnête.** À rang 128, la **PCA jointe reste le plafond** ; le seuil
KL ≤ 0,03 n'est **pas atteignable en raffinant la projection linéaire de rang
128**. Le sous-espace capte 95,5 % de l'énergie poolée clés+requêtes ; les 4,5 %
restants sont la borne structurelle qui se lit dans le KL. Franchir le seuil
demanderait soit un **rang plus élevé** (casse l'invariant tuile 128 o), soit une
**projection non-linéaire** (hors périmètre) — pas un meilleur réglage du levier
linéaire. Reproduction : `train_on_real_activations --dump train --joint --sgd
--sgd-lr 1e-6 --sgd-epochs 50` puis `offline_validation --dump test --weights
… --codec {mixed|mix3}` (dumps GPT-2 c6 WikiText-2, §3bis). Le drapeau `--sgd`
est conservé (il converge sur le synthétique, cf. `train_projection_reduces_score_loss`) ;
son échec sur activations réelles **est** le résultat.

**Ce que cela dit pour la suite.** Le cosinus de sortie (0,984) est déjà
excellent ; seul le KL reste au-dessus d'un seuil *pré-enregistré et strict*.
Savoir si ce KL de 0,055 dégrade réellement la perplexité d'un vrai modèle est
précisément la question de l'intégration llama.cpp (Phase 2, `docs/INTEGRATION.md`).

## 4. Bug amont trouvé pendant le portage

L'implémentation Rust de TurboQuant (`turboquant-core/src/qjl.rs`) avait une
grille mal dimensionnée : `increment = 0,5` produit 15 niveaux, puis le clamp
à 8 niveaux **écrasait toute valeur positive vers 0** (vérifié numériquement :
x = +0,25 / +0,5 / +1,0 → 0,0). Corrigé côté TurboQuant (grille 8 niveaux à
pas 1,0) ; côté SLHA, le test `tq3_positive_values_do_not_collapse` sert de
garde-fou de régression au portage.

## 5. Utilisation

```bash
# validation offline A/B
cargo run --release --example offline_validation -- --codec tq3
# comparatif des codecs
cargo run --release --example measure_learned
```

```rust
use scirust::attention::slha_v2::LatentCodec;
let tile = model.encode_with(&key, pos, /*warm=*/ false, LatentCodec::Tq3);
let score = tile.compute_score(&q_coarse, &q_sign); // SIMD dédié (AVX2/AVX-512/NEON)
```

Limites actuelles : TQ3 pur est **NO-GO mesuré comme codec HOT sur
activations réelles** (§3bis) — sur spectres réels, utiliser **MIX3**
(§3ter), qui garde le barreau de pagination à plan séparable au niveau de
qualité du mixte. TQ3 reste pertinent sur distributions plates. Décodage
SIMD : les cinq codecs ont leurs kernels dédiés (AVX2/AVX-512/NEON).

## 6. Licence

TurboQuant et SLHAv2 sont sous la **même double licence** : PolyForm
Noncommercial 1.0.0 (gratuit, non-commercial) + licence commerciale,
offerte exclusivement pour les déploiements **CCOS** dont les deux dépôts
sont les modules compagnons (voir [`LICENSING.md`](../LICENSING.md) ici et
dans le dépôt TurboQuant). Le portage TQ3 est une réutilisation interne du
même auteur (Copyright 2026 Tarek Zekriti) — les versions de TurboQuant
publiées avant l'alignement des licences restent sous leurs termes MIT/
Apache-2.0 d'origine.
