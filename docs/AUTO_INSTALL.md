# Installation augmentée et auto-branchement agent

Depuis le mode augmenté de `./install.sh`, SLHA v2 **détecte lui-même son
environnement** (agent IA, moteur LLM, matériel, toolchains) et peut
**s'auto-enregistrer auprès de l'agent IA** pour exposer ses outils MCP sans
aucune configuration manuelle.

## En une commande

```bash
./install.sh --auto --yes
```

Ce que fait `--auto` :

1. **Détecte l'environnement** (via [`scripts/detect_env.sh`](../scripts/detect_env.sh)) :
   l'agent IA (Claude Code / Claude Desktop / Cursor), le moteur LLM local
   (Ollama / vLLM / llama.cpp) et ses modèles, la GPU (NVIDIA + version CUDA),
   l'architecture CPU, le nombre de cœurs, la RAM, et les toolchains
   (rustc / cargo / python3 / nvcc / maturin).
2. **Construit et teste** le noyau (`scirust`), le serveur MCP (`slha-mcp`) et
   l'ABI C (`slha-c`), plus le binding Python si `python3-config` est présent.
3. **S'auto-enregistre** auprès de l'agent détecté (via
   [`scripts/register_mcp.sh`](../scripts/register_mcp.sh)) : le serveur
   `slha-mcp` devient un serveur MCP appelable par l'agent, sans qu'aucune
   commande `claude mcp add` manuelle ne soit nécessaire.
4. **Affiche un résumé** lisible de l'environnement détecté.

L'enregistrement est **best-effort** : si l'agent n'est pas utilisable (binaire
incomplet, aucun agent), l'installation **réussit quand même** et un
avertissement explique quoi faire.

## Modes et drapeaux

| Commande | Effet |
|---|---|
| `./install.sh` | Comportement historique : build + test des crates cœur, sans rien modifier d'autre. |
| `./install.sh --detect` | Imprime le rapport d'environnement et sort (aucune écriture). |
| `./install.sh --auto` | Mode augmenté : détecte, construit, teste, enregistre l'agent, résume. |
| `./install.sh --register-mcp` | Enregistre `slha-mcp` auprès de l'agent détecté (sans `--auto`). |
| `./install.sh --with-cuda` | Ajoute `slhav2-vram` avec la feature CUDA au build. |
| `./install.sh --yes` | Aucune confirmation interactive. |
| `./install.sh --skip-tests` | Saute les tests après le build. |
| `./install.sh --skip-python` | Ne construit pas le binding Python. |
| `./install.sh --skip-register` | Avec `--auto` : ne pas tenter l'enregistrement agent. |
| `./install.sh --mcp-bin <chemin>` | Binaire MCP à enregistrer (défaut : `target/release/slha-mcp`). |

Comme avant, un arbre Git modifié est refusé sauf `SLHA_ALLOW_DIRTY=1`, et tout
est construit avec `--locked`.

## Détection d'environnement (`scripts/detect_env.sh`)

Rapport structuré **JSON** (ou lisible avec `--human`) :

```json
{
  "agent": { "name": "claude-code", "cli": true, "desktop": false,
             "bin": "claude", "config_dir": "/home/me/.claude" },
  "llm":   { "engine": "ollama", "bin": "ollama",
             "url": "http://127.0.0.1:11434",
             "models": ["qwen3.6:35b", "..." ] },
  "hw":    { "gpu": "nvidia", "gpu_name": "NVIDIA Thor",
             "cuda_driver": "580.00", "arch": "aarch64",
             "cores": 14, "mem_bytes": 131881115648 },
  "toolchains": { "rustc": "...", "cargo": "...", "python3": "...",
                  "nvcc": "...", "maturin": "..." },
  "cuda_build": { "ptx_compiles": true },
  "paths": { "root": "/path/to/SLHAv2",
             "mcp_bin": "/path/to/SLHAv2/target/release/slha-mcp" }
}
```

Détection d'agent :

| Agent | Signal |
|---|---|
| **Claude Code** | binaire `claude` dans le PATH, ou `~/.claude`, ou `CLAUDE_CONFIG_DIR` |
| **Claude Desktop** | config `claude_desktop_config.json` (Linux `~/.config/Claude`, macOS `~/Library/Application Support/Claude`) |
| **Cursor** | binaire `cursor` |

Le moteur LLM est détecté par la présence d'`ollama` (avec la liste des
modèles si le serveur répond), de `vllm`, ou de `llama-cli`/`llama-server`
(llama.cpp). **Aucun appel réseau externe** : Ollama n'est sondé qu'en
localhost, et rien n'est téléchargé.

## Auto-enregistrement MCP (`scripts/register_mcp.sh`)

| Agent | Action |
|---|---|
| **Claude Code** | `claude mcp add slha -- <binaire>` (avec détection d'un enregistrement existant ; `--unregister` fait `claude mcp remove slha`). |
| **Claude Desktop** | insertion du bloc `mcpServers.slha` dans la config de l'application, avec sauvegarde `.bak`. |
| **Aucun / autre** | avec `--stdout`, imprime la config JSON à coller manuellement dans n'importe quel client MCP. |

Options : `--bin <chemin>`, `--stdout`, `--unregister`, `--yes`, `-h`.

Exemple de config générée (valable pour tout client MCP) :

```json
{
  "mcpServers": {
    "slha": {
      "command": "/chemin/vers/SLHAv2/target/release/slha-mcp"
    }
  }
}
```

Une fois enregistré, l'agent dispose des 5 outils `slha.*` (audit, explain,
compress, score, benchmark) — voir [`MCP.md`](MCP.md). Un test rapide :
demandez à l'agent d'« auditer le noyau SLHA ».

## Portée et honnêteté

- L'**intégration moteur LLM** (Ollama / vLLM / llama.cpp) est **détectée**
  mais **pas encore branchée** : SLHA v2 ne s'injecte pas dans un moteur
  d'inférence (c'est l'esquisse de [`INTEGRATION.md`](INTEGRATION.md)). La
  détection sert aujourd'hui au rapport d'environnement et prépare ce futur
  branchement.
- L'enregistrement MCP écrit **uniquement** dans la configuration de l'agent
  détecté (avec sauvegarde `.bak` pour Claude Desktop) ; `--stdout` permet de
  n'écrire nulle part.
- La détection lit des fichiers de configuration et exécute des commandes de
  sondage en **lecture seule** ; elle ne télécharge rien.
