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
une tuile 3 bits pure — un état de pagination CCOS *plus fin* que HOT→WARM
(candidat : HOT → TQ3-sans-correction → WARM → COLD). Non implémenté côté
`ccos` ; suivi dans la roadmap.

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
let score = tile.compute_score(&q_coarse, &q_sign); // chemin scalaire (comme NF4/mixte)
```

Limites actuelles : décodage **scalaire uniquement** (comme NF4 et mixte — le
décodage SIMD 3 bits est un suivi) ; l'état de pagination « correction lâchée »
n'est pas encore branché dans `ccos::ElasticKvCache`.
