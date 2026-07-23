.PHONY: all build test bench measure clean doc install help

# ─── SLHA v2 Makefile ─────────────────────────────────────────────
# Tapez 'make' ou 'make help' pour voir toutes les commandes.

all: build test

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installe Rust si nécessaire, clone et compile
	@bash install.sh

build: ## Compile le projet (mode release)
	cargo build --release --workspace --all-targets --all-features

dev: ## Compile en mode debug (plus rapide)
	cargo build --workspace --all-targets --all-features

test: ## Lance tous les tests
	cargo test --workspace --all-features

bench: ## Lance les micro-benchmarks
	cargo bench

measure: ## Benchmark complet (fidélité, débit, HOT/WARM)
	cargo run --example measure --release

measure-learned: ## Benchmark avec base apprise par PCA + INT4 groupé
	cargo run --example measure_learned --release

bench-fp16: ## Comparaison débit mémoire vs BF16
	cargo run --example bench_vs_fp16 --release

fidelity: ## Fidélité de la sortie softmax·V
	cargo run --example attention_fidelity --release

learn: ## Projection apprise vs PCA
	cargo run --example learn_projection --release

example: ## Lance l'exemple de base
	cargo run --example basic_usage

lint: ## Vérifie le style de code
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings

fmt: ## Formate le code
	cargo fmt --all

doc: ## Génère la documentation
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

clean: ## Nettoie les artefacts de compilation
	cargo clean

check: test lint bench ## Vérification complète (tests + lint + bench)
