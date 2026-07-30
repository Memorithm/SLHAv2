# slhav2-vram — gestion VRAM et scoring de tuiles SLHA côté GPU

Backend d'allocation VRAM et de scoring de tuiles SLHAv2 sur GPU, derrière la
feature opt-in `cuda` (chargement runtime de `libcuda` via des handles opaques ;
le PTX est produit par `nvcc` au build ou fourni via `SLHAV2_PTX_PATH`).

## Statut de vérification — à lire avant d'utiliser le backend CUDA

**Le backend CUDA n'est PAS vérifié à l'exécution.** Ce qui est vérifié, et où :

| Surface | Vérifiée ? | Comment |
|---|---|---|
| Compilation du chemin CUDA (feature `cuda`) | ✅ | CI : `cargo build/clippy/test --workspace --all-features` (sans GPU, le PTX est absent et `build.rs` l'annonce par un warning) |
| Tests CPU de la crate (traits, allocateur, fallback) | ✅ | CI : `cargo test --workspace --all-features` |
| **Exécution réelle sur GPU** — les 12 tests de `tests/cuda.rs` + `tests/cuda_unavailable.rs` | ❌ **jamais exécutée** | tous marqués `#[ignore = "requires an NVIDIA CUDA GPU"]` ; aucun runner GPU dans la CI |

Conséquences :

- Aucune garantie d'exactitude des kernels, des transferts ni de la gestion de
  contexte au-delà de ce que le type-checker et les tests CPU prouvent.
- Les blocs `unsafe` FFI de `src/backends/cuda.rs` sont audités sur documents
  (préconditions `// SAFETY:`), pas contre un driver réel.
- Sur une machine avec GPU NVIDIA + `nvcc`, exécuter la vérité terrain :
  `cargo test -p slhav2-vram --features cuda -- --ignored`.

Tant que cette commande n'a pas tourné sur un vrai GPU, ne traitez pas ce
backend comme validé. Ce statut est volontairement explicite : le projet
distingue « compile » de « vérifié » (cf. `docs/SUCCESS_CRITERIA.md`,
garde-fou « mesurer, pas affirmer »).

## Licence

Voir la licence du dépôt à la racine (`LICENSE`, `LICENSING.md`).
