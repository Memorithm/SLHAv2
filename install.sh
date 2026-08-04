#!/usr/bin/env bash
# Installation locale uniquement : examiner puis exécuter ./install.sh
#
# Modes :
#   ./install.sh                 # comportement historique : build + test des
#                                # crates cœur (scirust, slha-mcp, slha-c),
#                                # + python si disponible. Ne modifie rien
#                                # d'autre, n'écrit aucune config.
#
#   ./install.sh --detect        # imprime le rapport d'environnement
#                                # (scripts/detect_env.sh --human) et sort.
#
#   ./install.sh --auto          # mode augmenté : détecte l'environnement,
#                                # construit/teste, puis s'auto-enregistre
#                                # auprès de l'agent IA détecté (Claude Code /
#                                # Claude Desktop) et affiche le résumé.
#
# Drapeaux :
#   --register-mcp               # enregistre slha-mcp auprès de l'agent (même
#                                # sans --auto)
#   --with-cuda                  # active la feature CUDA pour slhav2-vram
#   --yes                        # ne pose aucune question (auto-confirme)
#   --skip-tests                 # saute les tests après le build
#   --skip-python                # ne construit pas le binding Python
#   --skip-register              # avec --auto : ne pas enregistrer l'agent
#   --mcp-bin <chemin>           # binaire MCP à enregistrer (défaut: release)
#
# Sortie : 0 si tout a réussi, non-0 sinon.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

die() { printf "ERREUR: %s\n" "$*" >&2; exit 1; }

# ── options ────────────────────────────────────────────────────────────────
MODE=minimal        # minimal | detect | auto
REGISTER=0
WITH_CUDA=0
YES=0
SKIP_TESTS=0
SKIP_PYTHON=0
SKIP_REGISTER=0
MCP_BIN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --detect) MODE=detect; shift ;;
    --auto)   MODE=auto; shift ;;
    --register-mcp) REGISTER=1; shift ;;
    --with-cuda) WITH_CUDA=1; shift ;;
    --yes) YES=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-python) SKIP_PYTHON=1; shift ;;
    --skip-register) SKIP_REGISTER=1; shift ;;
    --mcp-bin) MCP_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "option inconnue: $1" >&2; exit 2 ;;
  esac
done

# ── prérequis communs ───────────────────────────────────────────────────────
for cmd in git rustc cargo; do
    command -v "$cmd" >/dev/null || die "commande absente : $cmd"
done

[ -f Cargo.lock ] || die "Cargo.lock absent"
git rev-parse --is-inside-work-tree >/dev/null || die "dépôt Git requis"

# ── mode --detect : rapport d'environnement puis sortie ─────────────────────
# (opération en lecture seule : fonctionne même sur un arbre modifié)
if [ "$MODE" = "detect" ]; then
    if [ -x scripts/detect_env.sh ]; then
        exec ./scripts/detect_env.sh --human
    fi
    die "scripts/detect_env.sh introuvable"
fi

if [ "${SLHA_ALLOW_DIRTY:-0}" != "1" ] &&
   [ -n "$(git status --porcelain)" ]; then
    die "dépôt modifié ; utilisez explicitement SLHA_ALLOW_DIRTY=1"
fi

printf "Commit validé : %s\n" "$(git rev-parse HEAD)"

# ── build ───────────────────────────────────────────────────────────────────
CARGO_ARGS=(build --locked --release -p scirust -p slha-mcp -p slha-c)
if [ "$WITH_CUDA" = "1" ]; then
    CARGO_ARGS+=(-p slhav2-vram --features cuda)
fi

cargo "${CARGO_ARGS[@]}"

if [ "$SKIP_TESTS" != "1" ]; then
    cargo test --locked -p scirust -p slha-mcp -p slha-c
fi

if [ "$SKIP_PYTHON" != "1" ] &&
   command -v python3-config >/dev/null 2>&1 &&
   python3-config --embed --ldflags >/dev/null 2>&1; then
    cargo build --locked --release -p slha-python
    if [ "$SKIP_TESTS" != "1" ]; then
        cargo test --locked -p slha-python
    fi
fi

# ── enregistrement MCP ──────────────────────────────────────────────────────
REGISTER_WANTED=0
[ "$REGISTER" = "1" ] && REGISTER_WANTED=1
[ "$MODE" = "auto" ] && [ "$SKIP_REGISTER" != "1" ] && REGISTER_WANTED=1

if [ "$REGISTER_WANTED" = "1" ]; then
    if [ -x scripts/register_mcp.sh ]; then
        ARGS=()
        [ "$YES" = "1" ] && ARGS+=(--yes)
        if [ -n "$MCP_BIN" ]; then
            ARGS+=(--bin "$MCP_BIN")
        elif [ -x target/release/slha-mcp ]; then
            ARGS+=(--bin "$(pwd)/target/release/slha-mcp")
        fi
        ./scripts/register_mcp.sh "${ARGS[@]}" || {
            echo "AVERTISSEMENT: enregistrement MCP non abouti (voir ci-dessus)." >&2
        }
    else
        echo "AVERTISSEMENT: scripts/register_mcp.sh introuvable." >&2
    fi
fi

# ── résumé ──────────────────────────────────────────────────────────────────
printf "\nInstallation locale validée.\n"
if [ "$MODE" = "auto" ]; then
    echo "── Résumé de l'environnement ──"
    [ -x scripts/detect_env.sh ] && ./scripts/detect_env.sh --human || true
fi

[ "${SLHA_RUN_EXAMPLE:-0}" != "1" ] ||
    cargo run --locked -p scirust --example basic_usage
