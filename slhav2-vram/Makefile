.PHONY: all help ptx build test bench clean

# ─── slhav2-vram Makefile ──────────────────────────────────────

all: ptx build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

ptx: ## Compile CUDA kernel to PTX (requires nvcc)
	nvcc -ptx -arch=sm_89 -O3 --use_fast_math \
		kernels/lowrank_turboquant.cu \
		-o kernels/lowrank_turboquant.ptx

build: ## Compile the Rust crate (default features)
	cargo build

build-cuda: ## Compile with CUDA backend enabled
	cargo build --features cuda

test: ## Run all tests (default features)
	cargo test -- --nocapture

test-cuda: ## Run all tests with CUDA backend
	cargo test --features cuda -- --nocapture

bench: ## Run performance benchmarks (requires CUDA)
	cargo test --features cuda --test bench_rtx4060 -- --nocapture

clean: ## Clean build artifacts
	cargo clean
	rm -f kernels/lowrank_turboquant.ptx
