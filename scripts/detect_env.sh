#!/usr/bin/env bash
# SLHA v2 — détection d'environnement pour l'installation augmentée.
#
# Produit un rapport structuré (JSON sur stdout) des éléments que l'installeur
# augmenté consomme :
#   * l'agent IA disponible (Claude Code, Claude Desktop, Cursor, générique) ;
#   * le moteur LLM local (Ollama, vLLM, llama.cpp) et ses modèles ;
#   * le matériel (GPU NVIDIA/CUDA, CPU, RAM, cœurs) ;
#   * les toolchains (rustc, cargo, python3) ;
#   * l'état de build CUDA (nvcc + PTX du kernel).
#
# Usage:
#   ./scripts/detect_env.sh            # rapport JSON complet (stdout)
#   ./scripts/detect_env.sh --human    # rapport lisible (stdout)
#   ./scripts/detect_env.sh --json     # rapport JSON (défaut)
#
# Sortie JSON stable : { "agent": {...}, "llm": {...}, "hw": {...},
# "toolchains": {...}, "cuda_build": {...}, "paths": {...} }.
# Ne modifie rien ; ne lit que des fichiers de configuration et exécute des
# commandes de sondage en lecture seule.
set -euo pipefail

cd "$(dirname "$0")/.."

HUMAN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --human) HUMAN=1 ;;
    --json)  HUMAN=0 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "option inconnue: $1" >&2; exit 2 ;;
  esac
  shift
done

# ── helpers ────────────────────────────────────────────────────────────────

have() { command -v "$1" >/dev/null 2>&1; }

# json_str <value> — échappe une valeur pour JSON (gère les chaînes vides).
json_str() { printf '%s' "$1" | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read()))' 2>/dev/null \
  || printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"; }

# json_bool <0|1>
json_bool() { [ "$1" = "1" ] && printf true || printf false; }

# first_cmd <cmd...> — première commande trouvée dans PATH, sinon vide.
first_cmd() {
  local c
  for c in "$@"; do have "$c" && { printf '%s' "$c"; return 0; }; done
  return 0
}

# ── 1. agent IA ─────────────────────────────────────────────────────────────

detect_agent() {
  # Claude Code : config dans ~/.claude + binaire `claude`.
  local claude_code=0 claude_desktop=0 cursor=0 generic=0 name="none" bin=""
  [ -n "${CLAUDE_CONFIG_DIR:-}" ] && [ -d "${CLAUDE_CONFIG_DIR}" ] && claude_code=1
  [ -d "$HOME/.claude" ] && claude_code=1
  have claude && claude_code=1
  bin="$(first_cmd claude)"
  # Claude Desktop : config d'apps MCP.
  [ -f "$HOME/.config/Claude/claude_desktop_config.json" ] && claude_desktop=1
  [ -f "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ] && claude_desktop=1
  have cursor && cursor=1

  if [ "$claude_code" = "1" ]; then name="claude-code"; bin="${bin:-claude}";
  elif [ "$claude_desktop" = "1" ]; then name="claude-desktop";
  elif [ "$cursor" = "1" ]; then name="cursor";
  elif [ -n "${CLAUDE_MCP_SERVERS:-}" ]; then name="generic-mcp-env";
  else name="none"; fi

  printf '{"name":%s,"cli":%s,"desktop":%s,"cursor":%s,"bin":%s,"config_dir":%s}' \
    "$(json_str "$name")" "$(json_bool "$claude_code")" "$(json_bool "$claude_desktop")" \
    "$(json_bool "$cursor")" "$(json_str "$bin")" \
    "$(json_str "${CLAUDE_CONFIG_DIR:-$HOME/.claude}")"
}

# ── 2. moteur LLM local ─────────────────────────────────────────────────────

detect_llm() {
  local engine="none" bin="" models="[]" url=""
  if have ollama; then
    engine="ollama"; bin="ollama"; url="http://127.0.0.1:11434"
    # Lister les modèles si le serveur répond (lecture seule, timeout court).
    if command -v timeout >/dev/null 2>&1 && \
       timeout 3 ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -q .; then
      models="[$(timeout 3 ollama list 2>/dev/null | awk 'NR>1 {gsub(/^[[:space:]]+|[[:space:]]+$/,""); print "\""$1"\""}' | paste -sd, -)]"
    fi
  elif have vllm; then
    engine="vllm"; bin="vllm"
  elif have llama-cli || have llama-server || have main; then
    engine="llama.cpp"; bin="$(first_cmd llama-cli llama-server)"
  fi

  printf '{"engine":%s,"bin":%s,"url":%s,"models":%s}' \
    "$(json_str "$engine")" "$(json_str "$bin")" "$(json_str "$url")" "$models"
}

# ── 3. matériel ─────────────────────────────────────────────────────────────

detect_hw() {
  local gpu="none" cuda_ver="0" gpu_name="" mem_bytes=0
  if have nvidia-smi; then
    gpu="nvidia"
    cuda_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  fi
  if [ -r /proc/meminfo ]; then
    mem_bytes=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 ))
  fi
  local cores=0
  if [ -r /proc/cpuinfo ]; then cores=$(grep -c '^processor' /proc/cpuinfo || true); fi
  if [ "$cores" = "0" ] && have sysctl; then cores="$(sysctl -n hw.ncpu 2>/dev/null || echo 0)"; fi

  printf '{"gpu":%s,"gpu_name":%s,"cuda_driver":%s,"arch":%s,"cores":%d,"mem_bytes":%d}' \
    "$(json_str "$gpu")" "$(json_str "$gpu_name")" "$(json_str "$cuda_ver")" \
    "$(json_str "$(uname -m)")" "$cores" "$mem_bytes"
}

# ── 4. toolchains ───────────────────────────────────────────────────────────

detect_toolchains() {
  local rustc="$(rustc --version 2>/dev/null || echo none)"
  local cargo="$(cargo --version 2>/dev/null || echo none)"
  local python="$(python3 --version 2>/dev/null || echo none)"
  local nvcc="$(nvcc --version 2>/dev/null | tail -1 || echo none)"
  local maturin="$(python3 -m maturin --version 2>/dev/null || echo none)"

  printf '{"rustc":%s,"cargo":%s,"python3":%s,"nvcc":%s,"maturin":%s}' \
    "$(json_str "$rustc")" "$(json_str "$cargo")" "$(json_str "$python")" \
    "$(json_str "$nvcc")" "$(json_str "$maturin")"
}

# ── 5. build CUDA du noyau ─────────────────────────────────────────────────

detect_cuda_build() {
  local ptx="0" kernel=""
  if have nvcc && [ -f slhav2-vram/kernels/slha_score.cu ]; then ptx=1; fi
  if have nvcc; then
    local out
    out="$(nvcc -ptx -arch=sm_89 -O3 slhav2-vram/kernels/slha_score.cu -o /dev/null 2>&1)" \
      && ptx=1 || ptx=0
    [ -n "$out" ] && ptx=0
  fi
  printf '{"ptx_compiles":%s}' "$(json_bool "$ptx")"
}

# ── 6. chemins du produit ───────────────────────────────────────────────────

detect_paths() {
  printf '{"root":%s,"mcp_bin":%s}' \
    "$(json_str "$(pwd)")" \
    "$(json_str "$(pwd)/target/release/slha-mcp")"
}

# ── assemblage ──────────────────────────────────────────────────────────────

AGENT="$(detect_agent)"
LLM="$(detect_llm)"
HW="$(detect_hw)"
TOOLCHAINS="$(detect_toolchains)"
CUDA_BUILD="$(detect_cuda_build)"
PATHS="$(detect_paths)"

if [ "$HUMAN" = "1" ]; then
  python3 - "$AGENT" "$LLM" "$HW" "$TOOLCHAINS" "$CUDA_BUILD" "$PATHS" <<'PYEOF'
import json, sys
agent, llm, hw, tc, cuda, paths = (json.loads(x) for x in sys.argv[1:])
print("Agent IA      :", agent["name"] or "aucun détecté")
print("  cli         :", agent["cli"], " desktop:", agent["desktop"], " bin:", agent["bin"] or "-")
print("Moteur LLM    :", llm["engine"] or "aucun")
if llm["models"]:
    print("  modèles     :", ", ".join(llm["models"]))
print("Matériel      :", hw["arch"], "|", hw["cores"], "cœurs |", round(hw["mem_bytes"]/1e9,1), "Go RAM")
print("  GPU         :", hw["gpu"] if hw["gpu"]!="none" else "aucune", hw["gpu_name"])
print("Toolchains    :", tc["rustc"], "|", tc["python3"])
if tc["nvcc"] != "none":
    print("  CUDA        :", tc["nvcc"], "| PTX kernel:", cuda["ptx_compiles"])
print("Serveur MCP   :", paths["mcp_bin"])
PYEOF
else
  python3 - "$AGENT" "$LLM" "$HW" "$TOOLCHAINS" "$CUDA_BUILD" "$PATHS" <<'PYEOF'
import json, sys
agent, llm, hw, tc, cuda, paths = (json.loads(x) for x in sys.argv[1:])
print(json.dumps({
  "agent": agent, "llm": llm, "hw": hw,
  "toolchains": tc, "cuda_build": cuda, "paths": paths,
}, indent=2))
PYEOF
fi
