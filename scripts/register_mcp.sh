#!/usr/bin/env bash
# SLHA v2 — enregistrement automatique du serveur MCP `slha-mcp` auprès de
# l'agent IA détecté sur la machine.
#
# Agents supportés :
#   * Claude Code   — exécute `claude mcp add slha -- <binaire>` (le serveur est
#                     alors listé par `claude mcp list`) ;
#   * Claude Desktop — insère le bloc `mcpServers.slha` dans le fichier de
#                     configuration de l'application (avec sauvegarde .bak) ;
#   * autre / aucun  — mode `--stdout` : imprime la config JSON à coller
#                     manuellement (aucune écriture).
#
# Usage:
#   ./scripts/register_mcp.sh                 # enregistre dans l'agent détecté
#   ./scripts/register_mcp.sh --bin /path/to/slha-mcp
#   ./scripts/register_mcp.sh --stdout        # imprime la config, n'écrit rien
#   ./scripts/register_mcp.sh --unregister    # retire l'enregistrement
#   ./scripts/register_mcp.sh --yes           # ne demande pas confirmation
#
# Sortie : 0 si enregistré (ou déjà enregistré), 2 si aucun agent utilisable
# avec --stdout désactivé.
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="$(pwd)/target/release/slha-mcp"
STDOUT=0
UNREGISTER=0
YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --stdout) STDOUT=1; shift ;;
    --unregister) UNREGISTER=1; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "option inconnue: $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "ERREUR: binaire MCP introuvable: $BIN (construisez d'abord: cargo build --release -p slha-mcp)" >&2; exit 2; }

have() { command -v "$1" >/dev/null 2>&1; }

# ── détection de l'agent (mêmes règles que detect_env.sh) ──────────────────

AGENT="none"
have claude && AGENT="claude-code"
if [ "$AGENT" = "none" ] && { [ -f "$HOME/.config/Claude/claude_desktop_config.json" ] || \
     [ -f "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ]; }; then
  AGENT="claude-desktop"
fi

confirm() {
  [ "$YES" = "1" ] && return 0
  printf 'Enregistrer slha-mcp auprès de %s ? [y/N] ' "$AGENT" >&2
  read -r ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ]
}

# ── sortie générique (config JSON MCP standard) ─────────────────────────────

print_config() {
  cat <<EOF
{
  "mcpServers": {
    "slha": {
      "command": "$BIN"
    }
  }
}
EOF
}

# ── Claude Code ─────────────────────────────────────────────────────────────

register_claude_code() {
  if [ "$UNREGISTER" = "1" ]; then
    claude mcp remove slha 2>/dev/null || true
    echo "slha-mcp retiré de Claude Code."
    return 0
  fi
  # Le binaire `claude` doit être fonctionnel (install npm incomplète = échec).
  if have timeout; then
    if ! timeout 20 claude mcp list >/dev/null 2>&1; then
      echo "ERREUR: 'claude mcp list' échoue — binaire Claude Code non fonctionnel." >&2
      echo "  (install npm incomplète ? exécutez le postinstall de @anthropic-ai/claude-code)" >&2
      return 1
    fi
  elif ! claude mcp list >/dev/null 2>&1; then
    echo "ERREUR: 'claude mcp list' échoue — binaire Claude Code non fonctionnel." >&2
    return 1
  fi
  # Déjà enregistré ?
  if claude mcp list 2>/dev/null | grep -q '^slha[[:space:]]'; then
    echo "slha-mcp est déjà enregistré auprès de Claude Code (voir 'claude mcp list')."
    return 0
  fi
  if ! confirm; then echo "annulé."; return 1; fi
  if claude mcp add slha -- "$BIN"; then
    echo "✓ slha-mcp enregistré auprès de Claude Code."
    echo "  Vérifier: claude mcp list"
    echo "  Utiliser: demandez à l'agent d'« auditer le noyau SLHA »."
  else
    echo "ERREUR: 'claude mcp add' a échoué." >&2
    return 1
  fi
}

# ── Claude Desktop ──────────────────────────────────────────────────────────

desktop_config_path() {
  if [ -f "$HOME/.config/Claude/claude_desktop_config.json" ]; then
    printf '%s' "$HOME/.config/Claude/claude_desktop_config.json"
  elif [ -f "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ]; then
    printf '%s' "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
  else
    printf '%s' "$HOME/.config/Claude/claude_desktop_config.json"
  fi
}

register_claude_desktop() {
  local cfg; cfg="$(desktop_config_path)"
  if [ "$UNREGISTER" = "1" ]; then
    [ -f "$cfg" ] && python3 - "$cfg" <<'PYEOF'
import json, sys, os
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    sys.exit(0)
servers = d.get("mcpServers", {})
if "slha" in servers:
    del servers["slha"]
    d["mcpServers"] = servers
    json.dump(d, open(p, "w"), indent=2)
    print("slha-mcp retiré de Claude Desktop (%s)." % p)
else:
    print("slha-mcp absent de la config Claude Desktop.")
PYEOF
    return 0
  fi
  if ! confirm; then echo "annulé."; return 1; fi
  mkdir -p "$(dirname "$cfg")"
  [ -f "$cfg" ] && cp "$cfg" "$cfg.bak"
  python3 - "$cfg" "$BIN" <<'PYEOF'
import json, sys
p, bin_ = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p)) if __import__("os").path.exists(p) else {}
except Exception:
    d = {}
servers = d.setdefault("mcpServers", {})
servers["slha"] = {"command": bin_}
json.dump(d, open(p, "w"), indent=2)
print("✓ slha-mcp enregistré dans Claude Desktop (%s)." % p)
print("  Redémarrez Claude Desktop pour charger le serveur.")
PYEOF
}

# ── exécution ───────────────────────────────────────────────────────────────

if [ "$STDOUT" = "1" ]; then
  print_config
  exit 0
fi

case "$AGENT" in
  claude-code)    register_claude_code ;;
  claude-desktop) register_claude_desktop ;;
  *)
    echo "Aucun agent IA détecté (claude / Claude Desktop)." >&2
    echo "Config MCP à coller manuellement dans votre client MCP :" >&2
    print_config >&2
    exit 2
    ;;
esac
