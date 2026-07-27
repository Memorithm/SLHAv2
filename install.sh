#!/usr/bin/env bash
# Installation locale uniquement : examiner puis exécuter ./install.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

die() { printf "ERREUR: %s\n" "$*" >&2; exit 1; }

for cmd in git rustc cargo; do
    command -v "$cmd" >/dev/null || die "commande absente : $cmd"
done

[ -f Cargo.lock ] || die "Cargo.lock absent"
git rev-parse --is-inside-work-tree >/dev/null || die "dépôt Git requis"

if [ "${SLHA_ALLOW_DIRTY:-0}" != "1" ] &&
   [ -n "$(git status --porcelain)" ]; then
    die "dépôt modifié ; utilisez explicitement SLHA_ALLOW_DIRTY=1"
fi

printf "Commit validé : %s\n" "$(git rev-parse HEAD)"
cargo metadata --locked --no-deps --format-version 1 >/dev/null
cargo build --locked --release -p scirust -p slha-mcp -p slha-c
cargo test --locked -p scirust -p slha-mcp -p slha-c

if command -v python3-config >/dev/null 2>&1 &&
   python3-config --embed --ldflags >/dev/null 2>&1; then
    cargo build --locked --release -p slha-python
    cargo test --locked -p slha-python
fi

[ "${SLHA_RUN_EXAMPLE:-0}" != "1" ] ||
    cargo run --locked -p scirust --example basic_usage

printf "Installation locale validée.\n"
