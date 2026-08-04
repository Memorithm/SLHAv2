//! SLHA v2 — Sub-Low Rank Hybrid Attention micro-kernel (reference).
//!
//! Layout-aware tile + the fused float/binary score of eq. (2.3):
//!
//! ```text
//! score = <q_coarse, dequant(latent)>  +  lambda * (d_s - 2 * popcount(q_sign ^ B))
//!         \_______ continuous _______/     \_____________ binary (sign-LSH) ______/
//! ```
//!
//! This module is a thin facade over the split submodules:
//!
//! - layout constants, flags and codec geometry (`constants`);
//! - the [`SciRustSlhaTile`] type, per-codec dequant and the scalar score
//!   (`tile`);
//! - the AVX2/AVX-512/NEON kernels, the runtime dispatcher and the Hamming
//!   helper (`simd`);
//! - the per-codec quantisers (`codec`).
//!
//! ## What changed vs. the v1 listing (see spec §5.1)
//! - **No `read_volatile`.** The v1 reference read every element through
//!   `core::ptr::read_volatile`, which forbids LLVM from vectorising/reordering
//!   and thus *defeats* the stated performance goals. The hot path is now plain,
//!   auto-vectorisable scalar code over slices.
//! - **Signed INT4 (zero-point).** Dequantisation is `(nibble - 8) * scale`, so
//!   the latent base can represent **negative** values (real keys are signed).
//!   v1 used `nibble * scale`, clamping the base to `[0, 15]·scale >= 0`.
//! - **Safe API, no bogus `target_feature`.** `count_ones()` already lowers to
//!   `POPCNT` when the target supports it, with a portable fallback otherwise;
//!   no `unsafe` and no misleading `avx2` gate (the body has no AVX2 intrinsics).
//! - **Tile is exactly 128 bytes with zero padding** (see [`SciRustSlhaTile`]).

mod codec;
mod constants;
mod simd;
mod tile;

/// Compress a key vector and score a query against the tile.
///
/// ```
/// use scirust::attention::slha_v2::{quantize_latent, SciRustSlhaTile, D_C, RESIDUAL_WORDS};
///
/// // A 128-dim key, compressed to a 64-byte INT4 latent + scale.
/// let key = [0.5f32; D_C];
/// let (latent, scale) = quantize_latent(&key);
///
/// let tile = SciRustSlhaTile {
///     latent_kv: latent,
///     residual_bitmap: [0u64; RESIDUAL_WORDS],
///     scale,
///     dynamic_lambda: 0.5,
///     residual_sigma: 0.0,
///     token_id: 0,
///     position: 0,
///     head_id: 0,
///     flags: 0, // HOT
///     group_scales: [255u8; 8],
/// };
///
/// let q_coarse = [0.0f32; D_C];
/// let q_sign = [0u64; RESIDUAL_WORDS];
/// let score = tile.compute_score(&q_coarse, &q_sign);
/// assert!(score.is_finite());
/// ```
pub use codec::{
    quantize_latent, quantize_latent_grouped, quantize_latent_mix3, quantize_latent_mixed,
    quantize_latent_nf4, quantize_latent_tq3,
};
pub use constants::*;
pub use simd::hamming_distance;
pub use tile::{LatentCodec, SciRustSlhaTile};

// The x86-64 equivalence test exercises the private scalar and AVX-512
// implementations directly. Keep those helpers internal while making them
// visible to the child test module through its existing `use super::*`.
#[cfg(all(test, target_arch = "x86_64"))]
use simd::{hamming_scalar, hamming_vpopcntdq};

#[cfg(test)]
mod tests;
