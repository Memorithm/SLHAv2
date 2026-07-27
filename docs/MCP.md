# Connecter un agent / LLM à SLHA v2 (MCP)

`slha-mcp` est un serveur **MCP** (Model Context Protocol) qui expose le noyau
SLHA v2 et son auto-audit comme **outils appelables par un agent** (Claude Code,
Claude Desktop, ou tout client MCP). Il parle **JSON-RPC 2.0 délimité par
nouvelles lignes sur stdin/stdout** (le transport stdio de MCP).

> **Zéro dépendance externe** : tout le JSON-RPC réutilise `scirust::json`. Le
> serveur hérite donc de la propriété « sans dépendance » de SLHA v2.

## Démarrage rapide

```bash
# Construire le serveur (release recommandé : les chiffres de débit en dépendent)
cargo build --release -p slha-mcp
# Le binaire est à : target/release/slha-mcp
```

### Claude Code (CLI)

```bash
claude mcp add slha -- "$(pwd)/target/release/slha-mcp"
# ou, sans build préalable :
claude mcp add slha -- cargo run -q --release -p slha-mcp
```

### Claude Desktop / autre client (config JSON)

```json
{
  "mcpServers": {
    "slha": {
      "command": "/chemin/vers/SLHAv2/target/release/slha-mcp"
    }
  }
}
```

L'agent voit alors 5 outils. Demandez par exemple : *« audite le noyau SLHA »*,
*« quel débit de scoring sur cette machine ? »*, *« compresse ce vecteur clé »*.

## Outils exposés

| Outil | Arguments | Renvoie |
|---|---|---|
| `slha.audit` | — | Le rapport d'auto-audit **JSON** complet (invariants de tuile, équivalence SIMD≡scalaire *live*, features/cache, fidélité vs attention complète, budget CCOS, déterminisme). |
| `slha.explain` | — | Une explication en prose de SLHA v2 (tuile 128 o, codecs latents, score hybride, CCOS) que l'agent peut relayer. |
| `slha.compress` | `key`: 128 nombres ; `codec`? : `"int4"` \| `"grouped"` \| `"nf4"` \| `"mixed"` \| `"tq3"` (défaut `"int4"`) | Quantifie la clé dans le latent 64 o de la tuile avec le codec choisi et rapporte la compression (512 o FP32 → 128 o, ×4), l'échelle, les micro-échelles par groupe et le drapeau posé (`FLAG_NF4` / `FLAG_MIXED` / `FLAG_TQ3` ; 0 pour `int4`/`grouped`). |
| `slha.score` | `key`, `query`: 128 nombres | Construit une tuile depuis `key` et calcule le score SLHA pour `query` vs le produit scalaire exact (montre l'erreur de reconstruction INT4). |
| `slha.benchmark` | `n`? (défaut 200000) | Débit de scoring sur la machine hôte : `scores_per_sec`, `ns_per_score`, chemin SIMD dispatché. |

> `slha.audit` est l'outil phare : un agent (ou un humain) obtient en un appel
> l'état de santé vérifié du système, **les mêmes faits** que `slha-audit` en CLI.

Le paramètre `codec` de `slha.compress` reprend le mapping de l'exemple
`offline_validation` (`--codec grouped|nf4|mixed|tq3`), plus `int4` (INT4 à
échelle simple — le défaut, comportement historique de l'outil). `tq3` est le
port du codec TurboQuant (grille 3 bits + plan de correction de signe 1 bit
dans les mêmes 64 o) — mesures et trade-offs dans
[`TURBOQUANT.md`](TURBOQUANT.md).

## Protocole

- Transport : **stdio**, un message JSON-RPC par ligne (sans nouvelle ligne
  interne — le sérialiseur compact n'en produit pas).
- Version MCP annoncée : `2024-11-05`.
- Méthodes : `initialize`, `notifications/initialized` (silencieux), `ping`,
  `tools/list`, `tools/call`. Une méthode inconnue renvoie l'erreur JSON-RPC
  `-32601` ; une erreur d'**outil** est signalée par `isError: true` dans le
  résultat (conformément à MCP), pas par une erreur JSON-RPC.

## Exemple de session (brut)

```text
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"slha-mcp","version":"0.2.0"}}}

→ {"jsonrpc":"2.0","id":2,"method":"tools/list"}
← {"jsonrpc":"2.0","id":2,"result":{"tools":[ … 5 outils … ]}}

→ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"slha.benchmark","arguments":{"n":50000}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{ \"scores_per_sec\": …, \"dispatched_path\": \"avx512\", … }"}],"isError":false}}
```

## Portée (honnêteté)

`slha-mcp` rend le **noyau et l'audit** SLHA accessibles à un agent. Ce n'est
**pas** un greffon KV-cache pour un moteur d'inférence (llama.cpp / Ollama /
vLLM) — cette intégration-là reste une *esquisse* (voir
[`INTEGRATION.md`](INTEGRATION.md)). Les chiffres de `slha.benchmark` reflètent
le profil de build : utilisez `--release` pour des nombres représentatifs.
