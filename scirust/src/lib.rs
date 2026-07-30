//! SciRust — reference implementation of the SLHA v2 attention mechanism.
//!
//! SLHA v2 (Sub-Low Rank Hybrid Attention) splits the key/value representation
//! into two asymmetric components:
//!
//! 1. a **low-rank latent base** (`d_c` dims) stored as INT4, and
//! 2. a **1-bit sign-LSH residual** (`d_s` bits) capturing the high-frequency
//!    correction the low-rank base misses.
//!
//! The unnormalised attention score fuses a continuous dot product with a
//! binary (popcount-based) term — see [`attention::slha_v2`] and eq. (2.3) of
//! the spec.
//!
//! Correctness-first reference with a safe, testable API. The hot path has a
//! portable scalar fallback plus runtime-dispatched **AVX2** (x86_64) and
//! **NEON** (aarch64) kernels, each checked for equivalence against the scalar
//! path. (The v1 listing used `read_volatile`, which forbids auto-vectorisation;
//! that has been removed — see spec §5.1.)

// These are numeric kernels: indexing parallel arrays and matvec rows by
// position reads closer to the math than iterator-chain rewrites would.
#![allow(clippy::needless_range_loop)]

pub mod adapter;
pub mod attention;
pub mod audit;
pub mod ccos;
pub mod eventlog;
pub mod incoherence;
pub mod json;
pub mod learned;
pub mod linalg;
pub mod metrics;
pub mod norm_manifest;
pub mod numa;
pub mod ranking;
pub mod residual;
pub mod rng;
pub mod rope;
pub mod safety;
pub mod scenario;
pub mod weights;

pub use attention::slha_v2::{SciRustSlhaTile, D_C, D_S, LATENT_BYTES, RESIDUAL_WORDS};
