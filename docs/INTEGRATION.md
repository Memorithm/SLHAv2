# Intégrer SLHA v2 dans un moteur LLM existant

Ce guide explique comment brancher SLHA v2 dans **llama.cpp**, **Ollama**,
**vLLM** ou votre propre moteur d'inférence.

> **Statut.** La **Phase 2 llama.cpp** est engagée pour de vrai et
> **partiellement livrée** — voir [`integration/llama.cpp/`](../integration/llama.cpp/) :
> le **pont ABI C** (`slha-c` : `slha_weights_load` / `slha_encode_key` /
> `slha_decode_latent` / `slha_weights_free`) est **implémenté et testé**, une
> **perplexité de référence est mesurée** (Qwen2.5-0.5B Q8_0, WikiText-2 :
> **PPL 17,72**), et le **point d'insertion du patch est identifié dans le
> code** (`llama_kv_cache::cpy_k`, `ggml_map_custom1`). Ce qui reste (le shim
> ggml + projections par couche + mesure du Δppl) y est spécifié précisément.
>
> Les sections **Ollama / vLLM ci-dessous restent du pseudo-code** illustratif
> (les drapeaux `--kv-cache=slha` etc. n'existent pas) ; seules la Phase 2
> llama.cpp et la section « moteur Rust custom » reposent sur du code réel.

---

## Principe général

Tous les moteurs LLM ont une boucle d'attention qui fait :

```
pour chaque token généré :
    charger le KV-cache depuis la RAM
    calculer attention(Q, K, V)    ← c'est ici que SLHA intervient
    générer le token suivant
    ajouter le nouveau token au KV-cache
```

SLHA v2 remplace **le stockage et le calcul du KV-cache** pour que chaque token
occupe 128 octets au lieu de plusieurs kilo-octets.

---

## Intégration avec llama.cpp

### Architecture actuelle

```
llama.cpp
├── ggml (backend calcul)
├── llama (logique LLM)
│   └── llama_decode() → lit le KV-cache, calcule l'attention
└── gguf (format de modèle)
```

### Ce qu'il faut modifier

**Fichier : `llama.cpp` → fonction `llama_build_attention()`**

Avant :
```cpp
// Stockage standard : K et V en f16
struct llama_kv_cache {
    ggml_tensor * k_cache;  // ~2 Go pour 32k tokens
    ggml_tensor * v_cache;
};
```

Après (pseudo-code) :
```cpp
// Stockage SLHA : chaque token = 128 octets
struct llama_kv_cache_slha {
    uint8_t latent[64];        // base INT4 compressée
    uint64_t residual[4];      // bitmap de correction 1-bit
    float scale;
    float lambda;
    // ... autres métadonnées
};

// Dans la boucle d'attention :
float compute_attention_slha(query, token_idx) {
    auto tile = kv_cache[token_idx];
    
    // Déquantifier la base 4-bit → 128 dims
    float coarse = 0;
    for (int i = 0; i < 64; i++) {
        uint8_t byte = tile.latent[i];
        float v0 = ((byte & 0x0F) - 8) * tile.scale;
        float v1 = ((byte >> 4) - 8) * tile.scale;
        coarse += query.coarse[i*2] * v0;
        coarse += query.coarse[i*2+1] * v1;
    }
    
    // Correction binaire
    int pop = 0;
    for (int w = 0; w < 4; w++)
        pop += __builtin_popcountll(query.residual[w] ^ tile.residual[w]);
    float binary = tile.lambda * (256.0 - 2.0 * pop);
    
    return coarse + binary;
}
```

### Marche à suivre

1. **Forker llama.cpp** : `git clone https://github.com/ggerganov/llama.cpp`
2. **Ajouter SLHA comme sous-module** :
   ```bash
   git submodule add https://github.com/CHECKUPAUTO/SLHAv2 extern/SLHAv2
   ```
3. **Modifier `llama.cpp`** : remplacer le stockage KV par la structure SLHA
4. **Compiler** : `make LLAMA_SLHA=1`
5. **Tester** : lancer un modèle et comparer la qualité/perplexité

**Gain visé** (à mesurer sur un vrai modèle) : KV-cache plusieurs fois plus
petit (≈16× si l'on remplace un cache K+V FP16 complet). *Mesuré à ce jour*,
uniquement au niveau kernel : **~2,5× tokens/s** vs une clé bf16, à **2× moins
d'octets/token** (cf. `SLHAv2.md` §7.5).

---

## Intégration avec Ollama

Ollama utilise llama.cpp en interne. L'intégration est donc la même que
ci-dessus, mais appliquée à la version de llama.cpp embarquée dans Ollama.

```bash
# 1. Cloner le fork d'Ollama avec SLHA
git clone https://github.com/votre-fork/ollama.git
cd ollama

# 2. Appliquer le patch SLHA à llama.cpp embarqué
cd llm/llama.cpp
git submodule update --init
# ... appliquer les modifications SLHA ...

# 3. Rebuilder Ollama
cd ../..
go build -tags slha .

# 4. Lancer un modèle
./ollama run llama3.2 --kv-cache=slha
```

---

## Intégration avec vLLM

vLLM utilise PagedAttention. L'approche SLHA peut remplacer le stockage des
pages KV.

### Point d'entrée

**Fichier : `vllm/attention/ops/paged_attention.py`**

```python
# Avant : chaque bloc KV = N tokens × d_head × 2 (K+V) × 2 octets (f16)
# Après : chaque bloc KV = N tokens × 128 octets

class SLHABlock:
    def __init__(self, num_tokens: int):
        self.latent = torch.zeros(num_tokens, 64, dtype=torch.uint8)
        self.residual = torch.zeros(num_tokens, 4, dtype=torch.int64)
        self.scale = torch.zeros(num_tokens, dtype=torch.float32)
        self.lambda_ = torch.zeros(num_tokens, dtype=torch.float32)
    
    def compute_score(self, query_coarse, query_residual, token_idx):
        # Implémentation PyTorch du calcul SLHA
        byte = self.latent[token_idx]
        v0 = ((byte & 0x0F).float() - 8) * self.scale[token_idx]
        v1 = ((byte >> 4).float() - 8) * self.scale[token_idx]
        coarse = (query_coarse[::2] * v0).sum() + (query_coarse[1::2] * v1).sum()
        
        xor = query_residual ^ self.residual[token_idx]
        pop = xor.bit_count().sum()
        binary = self.lambda_[token_idx] * (256.0 - 2.0 * pop.float())
        
        return coarse + binary
```

---

## Intégration dans un moteur Rust custom

```rust
// Cargo.toml
// [dependencies]
// scirust = { git = "https://github.com/CHECKUPAUTO/SLHAv2" }

use scirust::attention::slha_v2;

pub struct SlhaKVcache {
    tiles: Vec<slha_v2::SciRustSlhaTile>,
}

impl SlhaKVcache {
    pub fn new(capacity: usize) -> Self {
        Self { tiles: Vec::with_capacity(capacity) }
    }

    pub fn insert(&mut self, key_vector: &[f32; 128], _value: &[f32]) {
        // Quantiser le vecteur de clé
        let (packed, scale) = slha_v2::quantize_latent(key_vector);
        
        let tile = slha_v2::SciRustSlhaTile {
            latent_kv: packed,
            residual_bitmap: [0u64; 4],
            scale,
            dynamic_lambda: 0.5, // calibré selon ρ
            residual_sigma: 0.0,
            token_id: self.tiles.len() as u32,
            position: self.tiles.len() as u32,
            head_id: 0,
            flags: slha_v2::FLAG_HOT,
            group_scales: [255u8; 8],
        };
        self.tiles.push(tile);
    }

    pub fn attention_score(&self, query: &[f32; 128], query_sign: &[u64; 4], idx: usize) -> f32 {
        self.tiles[idx].compute_score(query, query_sign)
    }

    /// HOT → WARM : libère le résidu 1-bit pour gagner 30% de mémoire
    pub fn page_out(&mut self, idx: usize) {
        self.tiles[idx].flags |= slha_v2::FLAG_WARM;
        self.tiles[idx].residual_bitmap = [0u64; 4];
    }
}
```

---

## Vérification de l'intégration

Après intégration, vérifiez ces points :

1. **Identité mathématique** : le score SLHA doit être égal au score scalaire
   (test d'équivalence déjà présent dans la crate)
2. **Débit** : comparer tokens/seconde avant/après sur un benchmark standard
3. **Perplexité** : mesurer la perplexité du modèle sur un jeu de test
   (ex: WikiText-2) — la différence doit être < 0.5
4. **Mémoire** : vérifier que le KV-cache occupe bien `N × 128` octets

---

## Support

- **Issues** : https://github.com/CHECKUPAUTO/SLHAv2/issues
- **Spécification** : [`SLHAv2.md`](../SLHAv2.md)
- **API Reference** : [`api.md`](api.md)
