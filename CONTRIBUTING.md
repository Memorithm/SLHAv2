# Contribuer à SLHA v2

Merci de votre intérêt ! Le dépôt est un **workspace Cargo** (le crate vit dans
`scirust/`). Toutes les commandes se lancent depuis la racine.

## Licence des contributions (CLA)

Le dépôt est en **double licence** (PolyForm Noncommercial 1.0.0 +
licence commerciale — voir [`LICENSING.md`](LICENSING.md)) ; SLHAv2 et
TurboQuant sont des modules compagnons de **CCOS**. Pour préserver ce
modèle, toute contribution externe n'est acceptée que sous un **accord de
licence de contributeur (CLA)** accordant au détenteur du copyright le
droit d'utiliser la contribution sous les deux licences. En ouvrant une
PR, signalez votre accord dans la description ; le CLA vous sera proposé
avant merge (contact : contact@checkupauto.fr).

## Avant d'ouvrir une PR

La CI exige que ces commandes passent — lancez-les en local :

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features   # suite complète : noyau + C + MCP + Python
cargo build --workspace --all-targets --all-features
cargo bench --workspace --all-features --no-run   # compile le graphe bench complet
```

Pour le chemin NEON (ARM), vérifiez la cross-compilation :

```bash
rustup target add aarch64-unknown-linux-gnu
# Même commande que la CI : on type-check (cargo check, pas build) scirust
# (chemin NEON) + slha-c (C-ABI, qui a déjà cassé sur aarch64) + slha-mcp
# contre la cible aarch64. `check` résout cfg(target_arch = "aarch64") sans
# link final — un `build` cross-linkerait slha-c (cdylib) et slha-mcp (bin)
# et demanderait un cross-linker aarch64 absent sur un hôte x86_64.
# slha-python (PyO3) est exclu — il nécessite les dev libs Python aarch64.
cargo check -p scirust -p slha-c -p slha-mcp --target aarch64-unknown-linux-gnu
```

## Principes du projet

- **Mesurer, pas affirmer.** Tout chiffre de performance/fidélité doit venir
  d'un test ou d'un exemple reproductible (graines fixes), pas d'une estimation.
  Les réserves d'honnêteté sont explicites (`FINDINGS.md`, `SLHAv2.md` §6–7).
- **La bibliothèque reste sans dépendance.** `criterion` est une dev-dependency
  (benches) uniquement.
- **Tout nouveau chemin SIMD** doit avoir un test d'équivalence `≡ scalaire`.
- **La doc doit décrire l'API réelle.** Pas d'API « plausible mais inexistante ».

## Style

`rustfmt` par défaut ; `clippy` sans warning (`-D warnings`). Les boucles
numériques peuvent utiliser l'indexation (`#![allow(clippy::needless_range_loop)]`
au niveau du crate).
