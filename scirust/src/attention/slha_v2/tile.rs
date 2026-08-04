//! The [`SciRustSlhaTile`] type: layout, per-codec dequant and the portable
//! scalar score path. SIMD kernels live in [`super::simd`].

use super::constants::*;

/// Which codec a tile's latent uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentCodec {
    /// Uniform INT4, single per-tile scale.
    Int4Single,
    /// Uniform INT4, per-group (MX) scales.
    Int4Grouped,
    /// NF4 (normal-float) codebook, per-group scales.
    Nf4,
    /// Mixed precision: top [`MIXED_HI_DIMS`] dims at 8-bit, next
    /// [`MIXED_LO_DIMS`] at 4-bit, tail dropped. Built for steep real-key
    /// spectra (outlier channels) that uniform INT4 cannot span. Assumes the
    /// latent is ordered by decreasing variance (PCA order).
    Mixed,
    /// TurboQuant TQ3: 3-bit symmetric grid plus a separable 1-bit
    /// sign-correction plane, per-group scales. Same worst-case resolution
    /// as INT4 in the same 64 bytes; the correction plane can be paged out
    /// independently (see [`FLAG_TQ3`]).
    Tq3,
    /// MIX3 synthesis: the mixed codec's 8-bit head (top [`MIXED_HI_DIMS`]
    /// dims) + a TQ3 body ([`MIXED_LO_DIMS`] dims at 3-bit with a separable
    /// correction plane), tail dropped. Near-mixed quality on steep real
    /// spectra with the TQ3 paging rung (see [`FLAG_MIX3`]).
    Mix3,
}

/// A single SLHA v2 context tile.
///
/// The field set makes the type **exactly 128 bytes with no padding**. Alignment
/// defaults to `align(64)` — two 64-byte cache lines, 64-aligned so it never
/// straddles a third line — which is correct and optimal on every 64-byte-line
/// part, i.e. **all our targets**: x86-64, and AArch64/Neoverse-V3AE (measured at
/// 64 B across L1d/L1i/L2 on a Jetson Thor AGX 128 — the "128" there is the 128 GB
/// unified CPU/GPU LPDDR5X memory, not the cache line).
///
/// On a **native** build whose host has genuine 128-byte lines (e.g. Apple
/// Silicon), [`build.rs`](../../build.rs) detects it and sets `cfg(cache_line_128)`,
/// raising the tile to `align(128)` so it occupies a single line. The size is
/// 128 bytes either way (a multiple of both alignments), so zero padding holds.
///
/// The 24 bytes that were *tail padding* in the v1 layout (104 useful bytes
/// rounded up by the alignment) are now spent on useful per-tile metadata.
///
/// Byte map (offsets): latent 0..64 | residual 64..96 | scale 96 | lambda 100 |
/// residual_sigma 104 | token_id 108 | position 112 | head_id 116 |
/// flags 118 | group_scales 120..128.
#[cfg_attr(cache_line_128, repr(C, align(128)))]
#[cfg_attr(not(cache_line_128), repr(C, align(64)))]
#[derive(Clone, Copy)]
pub struct SciRustSlhaTile {
    /// Latent base `h_KV` (128 dims) quantised to signed INT4. 64 bytes.
    pub latent_kv: [u8; LATENT_BYTES],
    /// Johnson–Lindenstrauss sign residual: 256 bits. 32 bytes.
    pub residual_bitmap: [u64; RESIDUAL_WORDS],
    /// INT4 dequantisation **global** scale; per-group refinement in `group_scales`.
    pub scale: f32,
    /// Binary-correction weight λ (eq. 3.2), calibrated per tile.
    pub dynamic_lambda: f32,
    /// Per-tile residual energy estimate σ_E (kept for λ recalibration).
    pub residual_sigma: f32,
    /// Token identifier (causal/event-log bookkeeping).
    pub token_id: u32,
    /// Sequence position.
    pub position: u32,
    /// Attention head id.
    pub head_id: u16,
    /// State flags (`FLAG_HOT` / `FLAG_WARM`).
    pub flags: u16,
    /// Per-group micro-scaling bytes: `effective_scale(g) = scale · gs[g]/255`.
    /// (Was reserved padding; now refines the INT4 latent — keeps the tile 128 B.)
    pub group_scales: [u8; N_GROUPS],
}

impl SciRustSlhaTile {
    /// True if the residual has been paged out (WARM mode).
    #[inline]
    pub fn is_warm(&self) -> bool {
        self.flags & FLAG_WARM != 0
    }

    /// True if the latent uses the NF4 codebook (else uniform INT4).
    #[inline]
    pub fn is_nf4(&self) -> bool {
        self.flags & FLAG_NF4 != 0
    }

    /// True if the latent uses the mixed-precision (8-bit head) layout.
    #[inline]
    pub fn is_mixed(&self) -> bool {
        self.flags & FLAG_MIXED != 0
    }

    /// True if the latent uses the TurboQuant TQ3 (3-bit + 1-bit) layout.
    #[inline]
    pub fn is_tq3(&self) -> bool {
        self.flags & FLAG_TQ3 != 0
    }

    /// True if the TQ3 correction plane has been paged out (see
    /// [`FLAG_TQ3_NOCORR`]): decode uses the bare 3-bit grid.
    #[inline]
    pub fn is_tq3_nocorr(&self) -> bool {
        self.flags & FLAG_TQ3_NOCORR != 0
    }

    /// True if the latent uses the MIX3 (mixed head × TQ3 body) layout.
    #[inline]
    pub fn is_mix3(&self) -> bool {
        self.flags & FLAG_MIX3 != 0
    }

    /// Size in bytes of this tile's separable correction plane (0 when the
    /// codec has none). This is what the CCOS correction-drop rung reclaims.
    #[inline]
    pub fn separable_corr_bytes(&self) -> usize {
        if self.is_tq3() {
            TQ3_CORR_BYTES
        } else if self.is_mix3() {
            MIX3_CORR_BYTES
        } else {
            0
        }
    }

    /// Effective dequant scale for dimension `d`: global scale × the dim's
    /// per-group micro-scale.
    #[inline]
    pub fn group_scale(&self, d: usize) -> f32 {
        self.scale * (self.group_scales[d / GROUP_DIM] as f32 / 255.0)
    }

    /// Dequantise one latent dimension `d` with its per-group scale, decoding
    /// the nibble via uniform INT4 (signed zero-point), the NF4 codebook, or
    /// the mixed-precision layout.
    #[inline]
    pub fn dequant_at(&self, d: usize) -> f32 {
        if self.is_mixed() {
            return self.dequant_at_mixed(d);
        }
        if self.is_tq3() {
            return self.dequant_at_tq3(d);
        }
        if self.is_mix3() {
            return self.dequant_at_mix3(d);
        }
        let byte = self.latent_kv[d >> 1];
        let nib = (if d & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as usize;
        let level = if self.is_nf4() {
            NF4_CODEBOOK[nib]
        } else {
            (nib as i32 - 8) as f32
        };
        level * self.group_scale(d)
    }

    /// Mixed layout: dims `0..MIXED_HI_DIMS` are signed bytes (zero-point 128)
    /// scaled by `group_scales[0]`; dims `MIXED_HI_DIMS..MIXED_DIMS` are
    /// nibbles in `GROUP_DIM`-wide groups scaled by `group_scales[1..]`; the
    /// dropped tail decodes to 0.
    #[inline]
    fn dequant_at_mixed(&self, d: usize) -> f32 {
        if d < MIXED_HI_DIMS {
            let level = self.latent_kv[d] as i32 - 128;
            level as f32 * (self.scale * self.group_scales[0] as f32 / 255.0)
        } else if d < MIXED_DIMS {
            let ld = d - MIXED_HI_DIMS;
            let byte = self.latent_kv[MIXED_HI_DIMS + (ld >> 1)];
            let nib = (if ld & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as i32;
            let g = 1 + ld / GROUP_DIM;
            (nib - 8) as f32 * (self.scale * self.group_scales[g] as f32 / 255.0)
        } else {
            0.0
        }
    }

    /// TQ3 layout: dim `d`'s 3-bit code is bits `[3d, 3d+3)` of the code
    /// plane; its correction sign is bit `d` of the correction plane. Decoded
    /// level = `(code − 3.5) ± TQ3_CORRECTION`, times the dim's group scale.
    /// With [`FLAG_TQ3_NOCORR`] the correction plane is considered freed and
    /// the bare grid level is returned.
    #[inline]
    fn dequant_at_tq3(&self, d: usize) -> f32 {
        let bit = 3 * d;
        let byte = bit >> 3;
        let shift = bit & 7;
        // A 3-bit field spans at most two bytes; the last code (d = 127)
        // ends exactly on the plane boundary, so guard the second read.
        let lo = u16::from(self.latent_kv[byte]);
        let hi = if byte + 1 < TQ3_CODE_BYTES {
            u16::from(self.latent_kv[byte + 1]) << 8
        } else {
            0
        };
        let code = ((lo | hi) >> shift) & 0x7;
        let mut level = code as f32 - TQ3_HALF_RANGE;
        if !self.is_tq3_nocorr() {
            let corr = (self.latent_kv[TQ3_CODE_BYTES + (d >> 3)] >> (d & 7)) & 1;
            let sign = if corr == 1 { 1.0 } else { -1.0 };
            level += sign * TQ3_CORRECTION;
        }
        level * self.group_scale(d)
    }

    /// MIX3 layout: dims `0..MIXED_HI_DIMS` decode exactly like the mixed
    /// head (signed bytes, `group_scales[0]`); dims
    /// `MIXED_HI_DIMS..MIXED_DIMS` decode like a TQ3 body — 3-bit code at
    /// bits `[3·ld, 3·ld+3)` of the plane starting at [`MIX3_CODES_OFF`],
    /// correction bit `ld` of the plane at [`MIX3_CORR_OFF`] (skipped with
    /// [`FLAG_TQ3_NOCORR`]) — scaled by `group_scales[1..]`; the dropped
    /// tail decodes to 0.
    #[inline]
    fn dequant_at_mix3(&self, d: usize) -> f32 {
        if d < MIXED_HI_DIMS {
            let level = self.latent_kv[d] as i32 - 128;
            return level as f32 * (self.scale * self.group_scales[0] as f32 / 255.0);
        }
        if d >= MIXED_DIMS {
            return 0.0;
        }
        let ld = d - MIXED_HI_DIMS;
        let bit = 3 * ld;
        let byte = MIX3_CODES_OFF + (bit >> 3);
        let shift = bit & 7;
        // A 3-bit field spans at most two bytes; the last code (ld = 111)
        // ends exactly on the plane boundary, so guard the second read.
        let lo = u16::from(self.latent_kv[byte]);
        let hi = if byte + 1 < MIX3_CORR_OFF {
            u16::from(self.latent_kv[byte + 1]) << 8
        } else {
            0
        };
        let code = ((lo | hi) >> shift) & 0x7;
        let mut level = code as f32 - TQ3_HALF_RANGE;
        if !self.is_tq3_nocorr() {
            let corr = (self.latent_kv[MIX3_CORR_OFF + (ld >> 3)] >> (ld & 7)) & 1;
            let sign = if corr == 1 { 1.0 } else { -1.0 };
            level += sign * TQ3_CORRECTION;
        }
        let g = 1 + ld / GROUP_DIM;
        level * (self.scale * self.group_scales[g] as f32 / 255.0)
    }

    /// Materialise the full dequantised latent vector (mostly for tests).
    pub fn dequant_latent(&self) -> [f32; D_C] {
        let mut out = [0.0f32; D_C];
        for (d, o) in out.iter_mut().enumerate() {
            *o = self.dequant_at(d);
        }
        out
    }

    /// Binary 1-bit correction: λ · (d_s − 2·popcount(q_sign ^ B)).
    /// popcount(XOR) is the Hamming distance; d_s − 2·Hamming is the signed dot
    /// product of the two ±1 sign vectors.
    #[inline]
    pub(super) fn residual_term(&self, q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        let hamming = super::hamming_distance(q_sign, &self.residual_bitmap);
        self.dynamic_lambda * (D_S as f32 - 2.0 * hamming as f32)
    }

    /// Portable scalar reference path.
    pub fn compute_score_scalar(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        let k = self.dequant_latent();
        let mut coarse = 0.0f32;
        for d in 0..D_C {
            coarse += q_coarse[d] * k[d];
        }
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }
}
