#!/usr/bin/env bash
# SLHA v2 — One-click installer
# Usage: curl -sSL https://raw.githubusercontent.com/CHECKUPAUTO/SLHAv2/master/install.sh | bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

banner() {
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║       SLHA v2 — Installeur rapide       ║"
    echo "  ║   Faites tourner une IA sans GPU         ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERREUR]${NC} $*"; }

# Pose une question même sous `curl | bash` (où stdin est le script lui-même,
# pas le clavier). Lit depuis le terminal s'il existe, sinon renvoie la valeur
# par défaut sans bloquer. Le prompt et la note vont sur stderr ; seule la
# réponse part sur stdout (pour la substitution de commande).
ask() {
    local prompt="$1" default="$2" reply=""
    # Probe in a subshell so a failed open neither prints nor exits the script.
    if (exec </dev/tty) 2>/dev/null; then
        read -rp "$prompt" reply </dev/tty || reply=""
    else
        printf '%s[non-interactif → défaut : %s]\n' "$prompt" "$default" >&2
    fi
    printf '%s' "${reply:-$default}"
}

banner

# ── 1. Vérifier ou installer Rust ──────────────────────────────────
if ! command -v rustc &>/dev/null; then
    info "Rust n'est pas installé — installation en cours..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    info "Rust installé : $(rustc --version)"
else
    info "Rust détecté : $(rustc --version)"
fi

# ── 2. Cloner le dépôt ─────────────────────────────────────────────
REPO="https://github.com/CHECKUPAUTO/SLHAv2.git"
DIR="SLHAv2"

if [ -d "$DIR" ]; then
    warn "Le dossier '$DIR' existe déjà."
    answer="$(ask "Le supprimer et re-cloner ? [o/N] " "N")"
    case "$answer" in
        o | O)
            rm -rf "$DIR"
            info "Clonage de $REPO..."
            git clone "$REPO" "$DIR"
            ;;
        *) info "Utilisation du dossier existant." ;;
    esac
else
    info "Clonage de $REPO..."
    git clone "$REPO" "$DIR"
fi

cd "$DIR"

# ── 3. Compiler ─────────────────────────────────────────────────────
info "Compilation du noyau, du serveur MCP et de l’ABI C..."
cargo build --release \
    -p scirust \
    -p slha-mcp \
    -p slha-c

# ── 4. Lancer les tests ─────────────────────────────────────────────
info "Lancement des tests..."
cargo test \
    -p scirust \
    -p slha-mcp \
    -p slha-c \
    2>&1 | tail -20

# Le binding Python nécessite une installation Python de développement capable
# de lier libpython. Il reste optionnel afin que l'installation du noyau,
# du serveur MCP et de l'ABI C soit portable et ne dépende pas de Python.
if command -v python3 >/dev/null 2>&1 \
    && command -v python3-config >/dev/null 2>&1 \
    && python3-config --embed --ldflags >/dev/null 2>&1
then
    info "Environnement Python de développement détecté."

    cargo build --release -p slha-python
    cargo test -p slha-python

    if command -v maturin >/dev/null 2>&1; then
        info "Construction de la roue Python avec Maturin..."

        (
            cd slha-python
            maturin build --release
        )
    else
        warn "Maturin absent : module Rust validé, roue Python non construite."
        warn "Commande facultative : cargo install maturin"
    fi
else
    warn "Binding Python ignoré : python3-config/libpython indisponible."
    warn "Le noyau, le serveur MCP et l'ABI C sont néanmoins validés."
fi

# ── 5. Premier essai ────────────────────────────────────────────────
echo ""
info "SLHA v2 est installé et prêt !"
echo ""
echo -e "  ${CYAN}Commandes utiles :${NC}"
echo "  cargo test -p scirust -p slha-mcp -p slha-c   # Suite portable sans Python"
echo "  cargo test --workspace --all-features          # Suite complète (Python dev requis)"
echo "  cargo run --example measure --release       # Benchmark complet"
echo "  cargo run --example basic_usage             # Exemple simple"
echo "  cargo bench                                 # Micro-benchmarks"
echo ""
echo -e "  ${CYAN}Documentation :${NC}"
echo "  docs/GETTING_STARTED.md    # Guide débutant"
echo "  docs/INTEGRATION.md        # Intégrer dans un projet"
echo "  SLHAv2.md                  # Spécification complète"
echo ""

# ── 6. Lancer l'exemple ─────────────────────────────────────────────
run="$(ask "Lancer l'exemple maintenant ? [O/n] " "O")"
case "$run" in
    n | N) info "Terminé. Lancez l'exemple quand vous voulez : cargo run --example basic_usage" ;;
    *)
        echo ""
        info "Lancement de l'exemple..."
        cargo run --example basic_usage
        ;;
esac
