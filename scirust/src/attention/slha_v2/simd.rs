//! SIMD scoring kernels and the runtime dispatcher.
//!
//! Every codec has a SIMD decode at every ISA level (uniform INT4 nibbles,
//! the NF4 codebook, the mixed 8-bit-head layout, TQ3 3-bit+correction and
//! MIX3 mixed-head × TQ3-body). All kernels are `#[target_feature]` methods
//! on [`SciRustSlhaTile`] guarded by runtime feature detection in
//! [`SciRustSlhaTile::compute_score`]; the scalar reference is
//! [`SciRustSlhaTile::compute_score_scalar`] (in [`super::tile`]).

use super::constants::*;
use super::tile::SciRustSlhaTile;

impl SciRustSlhaTile {
    /// Fused asymmetric attention score (eq. 2.3).
    ///
    /// `q_coarse` is `Q · W_up` in the latent space (`D_C` dims); `q_sign` is
    /// the packed sign of `Q · Zᵀ`. In WARM mode the binary term is dropped.
    ///
    /// Dispatches to an AVX-512/AVX2 (x86-64) or NEON (aarch64) path at
    /// runtime when available, else the portable scalar path. All paths yield
    /// the same result up to float reassociation.
    #[inline]
    pub fn compute_score(&self, q_coarse: &[f32; D_C], q_sign: &[u64; RESIDUAL_WORDS]) -> f32 {
        // Every codec has a SIMD decode at every ISA level: uniform-INT4
        // nibbles, the NF4 codebook, the mixed 8-bit-head layout, TQ3
        // (3-bit code + 1-bit correction) and MIX3 (mixed head × TQ3 body).
        // The codec checks mirror `dequant_at`'s flag precedence; hosts with
        // neither AVX level fall back to the portable scalar path.
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                // SAFETY: guarded by runtime feature detection.
                return unsafe {
                    if self.is_mixed() {
                        self.compute_score_avx512_mixed(q_coarse, q_sign)
                    } else if self.is_tq3() {
                        self.compute_score_avx512_tq3(q_coarse, q_sign)
                    } else if self.is_mix3() {
                        self.compute_score_avx512_mix3(q_coarse, q_sign)
                    } else if self.is_nf4() {
                        self.compute_score_avx512_nf4(q_coarse, q_sign)
                    } else {
                        self.compute_score_avx512(q_coarse, q_sign)
                    }
                };
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime feature detection.
                return unsafe {
                    if self.is_mixed() {
                        self.compute_score_avx2_mixed(q_coarse, q_sign)
                    } else if self.is_tq3() {
                        self.compute_score_avx2_tq3(q_coarse, q_sign)
                    } else if self.is_mix3() {
                        self.compute_score_avx2_mix3(q_coarse, q_sign)
                    } else if self.is_nf4() {
                        self.compute_score_avx2_nf4(q_coarse, q_sign)
                    } else {
                        self.compute_score_avx2(q_coarse, q_sign)
                    }
                };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64 — no runtime detection needed.
            // SAFETY: NEON is always available on this target.
            unsafe {
                if self.is_mixed() {
                    self.compute_score_neon_mixed(q_coarse, q_sign)
                } else if self.is_tq3() {
                    self.compute_score_neon_tq3(q_coarse, q_sign)
                } else if self.is_mix3() {
                    self.compute_score_neon_mix3(q_coarse, q_sign)
                } else if self.is_nf4() {
                    self.compute_score_neon_nf4(q_coarse, q_sign)
                } else {
                    self.compute_score_neon(q_coarse, q_sign)
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        self.compute_score_scalar(q_coarse, q_sign)
    }

    /// AVX2 path: vectorised INT4 dequant + dot for the coarse term.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm256_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `(nibble - 8) * group_scale` for 8 dims, multiply by q, accumulate.
        macro_rules! group_half {
            ($bytes:expr, $off:expr, $gs_v:expr) => {{
                let n = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($bytes));
                let v = _mm256_mul_ps($gs_v, _mm256_sub_ps(n, eight));
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs_v = _mm256_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            // Per-byte high nibble: shift 16-bit lanes, then mask each byte.
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            // Interleave so nibbles come out in dimension order.
            let d16 = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            group_half!(d16, base, gs_v);
            group_half!(_mm_srli_si128(d16, 8), base + 8, gs_v);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 TQ3 path: vectorised 3-bit + correction-bit dequant + dot for the
    /// coarse term of a [`FLAG_TQ3`] tile (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Decode strategy: 8 dims per step. Dim `8b+l` occupies bits
    /// `[3l, 3l+3)` of the 4-byte little-endian window at byte `3b`, so a
    /// broadcast of the window plus per-lane shifts `3l` (`vpsrlvd`) and a
    /// `& 7` mask denibbles the whole block; the matching correction bits are
    /// exactly byte `TQ3_CODE_BYTES + b`, extracted the same way with shifts
    /// `l` and `& 1`.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_tq3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
        let corr_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let seven = _mm256_set1_epi32(7);
        let one = _mm256_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // Fold the correction into the grid offset: level = code − 3.5 ±
        // 0.25 = code − 3.75 + corr·0.5 (all quarter-step values are exact
        // in f32, so this matches the scalar decode exactly per dim).
        let bias = _mm256_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm256_set1_ps(2.0 * TQ3_CORRECTION);
        let mut acc = _mm256_setzero_ps();
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 16 blocks × 8 dims (3 code bytes + 1 correction byte each) = 128
        // dims; one scale per 16-dim group = two consecutive blocks.
        for b in 0..D_C / 8 {
            let gs_v = _mm256_set1_ps(global * (self.group_scales[b / 2] as f32 * inv255));
            // 4-byte little-endian window holding all 8 code fields of this
            // block. The last block (b = 15) reads bytes 45..=48; byte 48 is
            // the first correction byte — in-bounds of `latent_kv`, and its
            // bits sit at window positions >= 24 > 3l + 2 for every lane
            // l <= 7, so the `& 7` mask after the per-lane shift keeps them
            // out of every decoded code.
            let window =
                u32::from_le_bytes([lk[3 * b], lk[3 * b + 1], lk[3 * b + 2], lk[3 * b + 3]]);
            let codes = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(window as i32), code_shifts),
                seven,
            );
            let mut level = _mm256_sub_ps(_mm256_cvtepi32_ps(codes), bias);
            if !nocorr {
                let cbyte = i32::from(lk[TQ3_CODE_BYTES + b]);
                let corr = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set1_epi32(cbyte), corr_shifts),
                    one,
                );
                level = _mm256_add_ps(level, _mm256_mul_ps(_mm256_cvtepi32_ps(corr), corr_step));
            }
            let v = _mm256_mul_ps(level, gs_v);
            let qv = _mm256_loadu_ps(q.add(8 * b));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 NF4 path: vectorised codebook dequant + dot for the coarse term
    /// of a [`FLAG_NF4`] tile.
    ///
    /// Decode strategy: nibbles unpack to i32 lanes exactly like
    /// [`Self::compute_score_avx2`]; the 16-entry [`NF4_CODEBOOK`] lookup is
    /// two `_mm256_permutevar8x32_ps` gathers (one per codebook half, both
    /// indexed by the low 3 nibble bits) blended on nibble bit 3.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_nf4(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let cb_lo = _mm256_loadu_ps(NF4_CODEBOOK.as_ptr());
        let cb_hi = _mm256_loadu_ps(NF4_CODEBOOK.as_ptr().add(8));
        let seven = _mm256_set1_epi32(7);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `NF4_CODEBOOK[nib] * group_scale` for 8 dims, multiply by
        // q, accumulate. `permutevar8x32` indexes with the low 3 bits only,
        // so both halves are gathered and nibble bit 3 picks the right one.
        macro_rules! group_half {
            ($bytes:expr, $off:expr, $gs_v:expr) => {{
                let nib = _mm256_cvtepu8_epi32($bytes);
                let lo_v = _mm256_permutevar8x32_ps(cb_lo, nib);
                let hi_v = _mm256_permutevar8x32_ps(cb_hi, nib);
                let pick_hi = _mm256_castsi256_ps(_mm256_cmpgt_epi32(nib, seven));
                let level = _mm256_blendv_ps(lo_v, hi_v, pick_hi);
                let v = _mm256_mul_ps($gs_v, level);
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs_v = _mm256_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            group_half!(d16, base, gs_v);
            group_half!(_mm_srli_si128(d16, 8), base + 8, gs_v);
        }

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 mixed-precision path: vectorised dequant + dot for the coarse
    /// term of a [`FLAG_MIXED`] tile.
    ///
    /// Decode strategy: the 8-bit head (dims `0..MIXED_HI_DIMS`) is a single
    /// 8-lane block — bytes to f32 minus the 128 zero-point, times the head
    /// scale `gs[0]`; the 4-bit body (dims `MIXED_HI_DIMS..MIXED_DIMS`) is
    /// [`Self::compute_score_avx2`]'s denibbling shifted 8 bytes/8 dims with
    /// group scales `gs[1..]`; the dropped tail (dims `MIXED_DIMS..D_C`)
    /// decodes to 0 and contributes nothing.
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_mixed(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm256_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS, one signed byte each
        // (zero-point 128), scaled by gs[0].
        let hs_v = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            latent as *const __m128i,
        )));
        let hv = _mm256_mul_ps(hs_v, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm256_mul_ps(hv, _mm256_loadu_ps(q));

        // Dequant `(nibble - 8) * group_scale` for 8 dims, multiply by q, accumulate.
        macro_rules! group_half {
            ($bytes:expr, $off:expr, $gs_v:expr) => {{
                let n = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32($bytes));
                let v = _mm256_mul_ps($gs_v, _mm256_sub_ps(n, eight));
                let qv = _mm256_loadu_ps(q.add($off));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
            }};
        }

        // 4-bit body: 7 groups × 8 bytes = 112 dims after the head; one
        // scale byte per 16-dim group (gs[1..]).
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs_v = _mm256_set1_ps(global * (self.group_scales[1 + g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(MIXED_HI_DIMS + g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // dims base..base+15
            group_half!(d16, base, gs_v);
            group_half!(_mm_srli_si128(d16, 8), base + 8, gs_v);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX2 MIX3 path: vectorised dequant + dot for the coarse term of a
    /// [`FLAG_MIX3`] tile (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Decode strategy: the 8-bit head is the mixed kernel's 8-lane block;
    /// the TQ3 body reuses `Self::compute_score_avx2_tq3`'s
    /// 8-dims-per-3-bytes window trick with every byte offset shifted to
    /// the MIX3 planes ([`MIX3_CODES_OFF`] / [`MIX3_CORR_OFF`]).
    ///
    /// # Safety
    /// The `avx2` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_score_avx2_mix3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
        let corr_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let seven = _mm256_set1_epi32(7);
        let one = _mm256_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = _mm256_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm256_set1_ps(2.0 * TQ3_CORRECTION);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS, exactly the mixed kernel's block.
        let hs_v = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            lk.as_ptr() as *const __m128i
        )));
        let hv = _mm256_mul_ps(hs_v, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm256_mul_ps(hv, _mm256_loadu_ps(q));

        // TQ3 body: 14 blocks × 8 dims (3 code bytes + 1 correction byte
        // each) = 112 dims after the head; one scale per 16-dim group = two
        // consecutive blocks (gs[1..]).
        for b in 0..MIXED_LO_DIMS / 8 {
            let gs_v = _mm256_set1_ps(global * (self.group_scales[1 + b / 2] as f32 * inv255));
            // 4-byte little-endian window at MIX3_CODES_OFF + 3b holding all
            // 8 code fields of this block. The last block (b = 13) reads
            // bytes 47..=50; byte 50 is the first correction byte
            // (MIX3_CORR_OFF) — in-bounds of `latent_kv`, and its bits sit
            // at window positions >= 24 > 3l + 2 for every lane l <= 7, so
            // the `& 7` mask after the per-lane shift keeps them out of
            // every decoded code.
            let o = MIX3_CODES_OFF + 3 * b;
            let window = u32::from_le_bytes([lk[o], lk[o + 1], lk[o + 2], lk[o + 3]]);
            let codes = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(window as i32), code_shifts),
                seven,
            );
            let mut level = _mm256_sub_ps(_mm256_cvtepi32_ps(codes), bias);
            if !nocorr {
                let cbyte = i32::from(lk[MIX3_CORR_OFF + b]);
                let corr = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set1_epi32(cbyte), corr_shifts),
                    one,
                );
                level = _mm256_add_ps(level, _mm256_mul_ps(_mm256_cvtepi32_ps(corr), corr_step));
            }
            let v = _mm256_mul_ps(level, gs_v);
            let qv = _mm256_loadu_ps(q.add(MIXED_HI_DIMS + 8 * b));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, qv));
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let coarse: f32 = tmp.iter().sum();

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 path: one 16-wide FMA per group (16 latent dims).
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm512_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm512_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // One group (16 dims = 8 bytes) per 16-wide fused multiply-add.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // 16 bytes = dims base..base+15
            let n = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(d16));
            let v = _mm512_mul_ps(gs, _mm512_sub_ps(n, eight));
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 TQ3 path: one 16-wide FMA per group for a [`FLAG_TQ3`] tile
    /// (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Same decode strategy as `Self::compute_score_avx2_tq3`, two 8-dim
    /// blocks per iteration: lanes 0..8 shift the 4-byte window at byte `6g`,
    /// lanes 8..16 the window at byte `6g + 3`, each by `3·(l mod 8)`; the
    /// group's 16 correction bits are the `u16` at `TQ3_CODE_BYTES + 2g`,
    /// shifted per lane by `l`.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_tq3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        #[rustfmt::skip]
        let code_shifts = _mm512_setr_epi32(
            0, 3, 6, 9, 12, 15, 18, 21,
            0, 3, 6, 9, 12, 15, 18, 21,
        );
        #[rustfmt::skip]
        let corr_shifts = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        );
        let seven = _mm512_set1_epi32(7);
        let one = _mm512_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = _mm512_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm512_set1_ps(2.0 * TQ3_CORRECTION);
        let mut acc = _mm512_setzero_ps();
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // One group (16 dims = 6 code bytes + 2 correction bytes) per
        // 16-wide fused multiply-add.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            // Two 4-byte little-endian windows: dims base..base+8 live at
            // bits [3l, 3l+3) of w0 (byte 6g), dims base+8..base+16 at the
            // same offsets of w1 (byte 6g+3). For g = 7, w1 reads bytes
            // 45..=48; byte 48 is the first correction byte — in-bounds of
            // `latent_kv`, and its bits sit at window positions >= 24, which
            // the shift-then-`& 7` can never bring into a decoded code.
            let w0 = u32::from_le_bytes([lk[6 * g], lk[6 * g + 1], lk[6 * g + 2], lk[6 * g + 3]]);
            let w1 =
                u32::from_le_bytes([lk[6 * g + 3], lk[6 * g + 4], lk[6 * g + 5], lk[6 * g + 6]]);
            let windows = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi32(w0 as i32)),
                _mm256_set1_epi32(w1 as i32),
                1,
            );
            let codes = _mm512_and_epi32(_mm512_srlv_epi32(windows, code_shifts), seven);
            let mut level = _mm512_sub_ps(_mm512_cvtepi32_ps(codes), bias);
            if !nocorr {
                let c16 = i32::from(lk[TQ3_CODE_BYTES + 2 * g])
                    | (i32::from(lk[TQ3_CODE_BYTES + 2 * g + 1]) << 8);
                let corr =
                    _mm512_and_epi32(_mm512_srlv_epi32(_mm512_set1_epi32(c16), corr_shifts), one);
                level = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr), corr_step, level);
            }
            let v = _mm512_mul_ps(level, gs);
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 NF4 path: one 16-wide FMA per group for a [`FLAG_NF4`] tile.
    ///
    /// Decode strategy: same denibbling as [`Self::compute_score_avx512`];
    /// the 16-entry [`NF4_CODEBOOK`] fits a single 512-bit register, so the
    /// codebook lookup is one `_mm512_permutexvar_ps` per group.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_nf4(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let codebook = _mm512_loadu_ps(NF4_CODEBOOK.as_ptr());
        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut acc = _mm512_setzero_ps();
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // One group (16 dims = 8 bytes) per 16-wide fused multiply-add.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi); // 16 bytes = dims base..base+15
            let level = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(d16), codebook);
            let v = _mm512_mul_ps(gs, level);
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 mixed-precision path: one 16-wide FMA per body group for a
    /// [`FLAG_MIXED`] tile.
    ///
    /// Same decode strategy as `Self::compute_score_avx2_mixed`: the
    /// 8-bit head is one 8-lane block (zero-extended into the 16-wide
    /// accumulator), the 4-bit body is [`Self::compute_score_avx512`]'s
    /// denibbling shifted 8 bytes/8 dims with group scales `gs[1..]`, and
    /// the dropped tail contributes nothing.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_mixed(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = _mm512_set1_ps(8.0);
        let nibble_mask = _mm_set1_epi8(0x0F);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS as one 8-lane block (zero-point
        // 128, scale gs[0]), zero-extended into the 16-wide accumulator.
        let hs = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            latent as *const __m128i,
        )));
        let hv = _mm256_mul_ps(hs, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm512_zextps256_ps512(_mm256_mul_ps(hv, _mm256_loadu_ps(q)));

        // 4-bit body: one group (16 dims = 8 bytes after the head) per
        // 16-wide fused multiply-add, scales gs[1..].
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[1 + g] as f32 * inv255));
            let packed = _mm_loadl_epi64(latent.add(MIXED_HI_DIMS + g * 8) as *const __m128i);
            let lo = _mm_and_si128(packed, nibble_mask);
            let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
            let d16 = _mm_unpacklo_epi8(lo, hi);
            let n = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(d16));
            let v = _mm512_mul_ps(gs, _mm512_sub_ps(n, eight));
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// AVX-512 MIX3 path: one 16-wide FMA per body group for a
    /// [`FLAG_MIX3`] tile (honours [`FLAG_TQ3_NOCORR`]).
    ///
    /// Same decode strategy as [`Self::compute_score_avx512_tq3`] for the
    /// body — two 4-byte windows per group, shifted to the MIX3 planes
    /// ([`MIX3_CODES_OFF`] / [`MIX3_CORR_OFF`]) — with the mixed kernel's
    /// 8-lane head zero-extended into the 16-wide accumulator.
    ///
    /// # Safety
    /// The `avx512f` target feature must be available. The public
    /// [`Self::compute_score`] dispatcher guarantees this via runtime detection;
    /// `pub` so benchmarks can target this path explicitly.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn compute_score_avx512_mix3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::x86_64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        #[rustfmt::skip]
        let code_shifts = _mm512_setr_epi32(
            0, 3, 6, 9, 12, 15, 18, 21,
            0, 3, 6, 9, 12, 15, 18, 21,
        );
        #[rustfmt::skip]
        let corr_shifts = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        );
        let seven = _mm512_set1_epi32(7);
        let one = _mm512_set1_epi32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = _mm512_set1_ps(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = _mm512_set1_ps(2.0 * TQ3_CORRECTION);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 8-bit head: same 8-lane block as the mixed kernel, zero-extended
        // into the 16-wide accumulator.
        let hs = _mm256_set1_ps(global * (self.group_scales[0] as f32 * inv255));
        let head = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            lk.as_ptr() as *const __m128i
        )));
        let hv = _mm256_mul_ps(hs, _mm256_sub_ps(head, _mm256_set1_ps(128.0)));
        let mut acc = _mm512_zextps256_ps512(_mm256_mul_ps(hv, _mm256_loadu_ps(q)));

        // TQ3 body: one group (16 dims = 6 code bytes + 2 correction bytes
        // after the head) per 16-wide fused multiply-add, scales gs[1..].
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs = _mm512_set1_ps(global * (self.group_scales[1 + g] as f32 * inv255));
            // Two 4-byte little-endian windows on the MIX3 code plane: body
            // dims base..base+8 live at bits [3l, 3l+3) of w0 (byte
            // MIX3_CODES_OFF + 6g), dims base+8..base+16 at the same offsets
            // of w1 (3 bytes later). For g = 6, w1 reads bytes 47..=50; byte
            // 50 is the first correction byte (MIX3_CORR_OFF) — in-bounds of
            // `latent_kv`, and its bits sit at window positions >= 24, which
            // the shift-then-`& 7` can never bring into a decoded code.
            let o = MIX3_CODES_OFF + 6 * g;
            let w0 = u32::from_le_bytes([lk[o], lk[o + 1], lk[o + 2], lk[o + 3]]);
            let w1 = u32::from_le_bytes([lk[o + 3], lk[o + 4], lk[o + 5], lk[o + 6]]);
            let windows = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi32(w0 as i32)),
                _mm256_set1_epi32(w1 as i32),
                1,
            );
            let codes = _mm512_and_epi32(_mm512_srlv_epi32(windows, code_shifts), seven);
            let mut level = _mm512_sub_ps(_mm512_cvtepi32_ps(codes), bias);
            if !nocorr {
                let c16 = i32::from(lk[MIX3_CORR_OFF + 2 * g])
                    | (i32::from(lk[MIX3_CORR_OFF + 2 * g + 1]) << 8);
                let corr =
                    _mm512_and_epi32(_mm512_srlv_epi32(_mm512_set1_epi32(c16), corr_shifts), one);
                level = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corr), corr_step, level);
            }
            let v = _mm512_mul_ps(level, gs);
            let qv = _mm512_loadu_ps(q.add(base));
            acc = _mm512_fmadd_ps(v, qv, acc);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.
        let coarse = _mm512_reduce_add_ps(acc);

        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON path (aarch64): vectorised INT4 dequant + dot for the coarse term.
    /// NEON is baseline on aarch64, so the dispatcher calls this unconditionally.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = vdupq_n_f32(8.0);
        let mask = vdup_n_u8(0x0F);
        let mut acc = vdupq_n_f32(0.0);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Dequant `(n - 8) * group_scale` for a 4-lane chunk, then FMA with q.
        macro_rules! quad {
            ($n4:expr, $off:expr, $gs:expr) => {{
                let v = vmulq_f32(vsubq_f32($n4, eight), $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = vdupq_n_f32(global * (self.group_scales[g] as f32 * inv255));
            let packed = vld1_u8(latent.add(g * 8)); // 8 bytes
            let lo = vand_u8(packed, mask);
            let hi = vshr_n_u8::<4>(packed);
            // Interleave so nibbles come out in dimension order.
            let d_lo = vzip1_u8(lo, hi); // dims base..base+7
            let d_hi = vzip2_u8(lo, hi); // dims base+8..base+15

            let w_lo = vmovl_u8(d_lo); // u16×8
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_lo))), base, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_lo))), base + 4, gs);

            let w_hi = vmovl_u8(d_hi);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_hi))), base + 8, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_hi))), base + 12, gs);
        }

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON TQ3 path (aarch64): vectorised 3-bit + correction-bit dequant +
    /// dot for the coarse term of a [`FLAG_TQ3`] tile (honours
    /// [`FLAG_TQ3_NOCORR`]). The dispatcher calls this unconditionally on
    /// aarch64, like [`Self::compute_score_neon`].
    ///
    /// Same decode strategy as `Self::compute_score_avx2_tq3`, split into
    /// two 4-lane quads per 8-dim block: `vshlq_u32` with negative per-lane
    /// counts is the NEON variable right-shift.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_tq3_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_tq3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        // Negative counts = right shifts: code fields at bits 3l, correction
        // bits at bit l, for lanes l of the low/high quad of an 8-dim block.
        const CODE_SH_LO: [i32; 4] = [0, -3, -6, -9];
        const CODE_SH_HI: [i32; 4] = [-12, -15, -18, -21];
        const CORR_SH_LO: [i32; 4] = [0, -1, -2, -3];
        const CORR_SH_HI: [i32; 4] = [-4, -5, -6, -7];

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_sh_lo = vld1q_s32(CODE_SH_LO.as_ptr());
        let code_sh_hi = vld1q_s32(CODE_SH_HI.as_ptr());
        let corr_sh_lo = vld1q_s32(CORR_SH_LO.as_ptr());
        let corr_sh_hi = vld1q_s32(CORR_SH_HI.as_ptr());
        let seven = vdupq_n_u32(7);
        let one = vdupq_n_u32(1);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = vdupq_n_f32(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = vdupq_n_f32(2.0 * TQ3_CORRECTION);
        let mut acc = vdupq_n_f32(0.0);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // Decode 4 dims from the block's window/correction byte, FMA with q.
        macro_rules! quad {
            ($window:expr, $cbyte:expr, $code_sh:expr, $corr_sh:expr, $off:expr, $gs:expr) => {{
                let codes = vandq_u32(vshlq_u32(vdupq_n_u32($window), $code_sh), seven);
                let mut level = vsubq_f32(vcvtq_f32_u32(codes), bias);
                if !nocorr {
                    let corr = vandq_u32(vshlq_u32(vdupq_n_u32($cbyte), $corr_sh), one);
                    level = vfmaq_f32(level, vcvtq_f32_u32(corr), corr_step);
                }
                let v = vmulq_f32(level, $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 16 blocks × 8 dims (3 code bytes + 1 correction byte each) = 128
        // dims; one scale per 16-dim group = two consecutive blocks.
        for b in 0..D_C / 8 {
            let gs = vdupq_n_f32(global * (self.group_scales[b / 2] as f32 * inv255));
            // 4-byte little-endian window holding all 8 code fields of this
            // block. The last block (b = 15) reads bytes 45..=48; byte 48 is
            // the first correction byte — in-bounds of `latent_kv`, and its
            // bits sit at window positions >= 24 > 3l + 2 for every lane
            // l <= 7, so the `& 7` mask after the per-lane shift keeps them
            // out of every decoded code.
            let window =
                u32::from_le_bytes([lk[3 * b], lk[3 * b + 1], lk[3 * b + 2], lk[3 * b + 3]]);
            let cbyte = u32::from(lk[TQ3_CODE_BYTES + b]);
            quad!(window, cbyte, code_sh_lo, corr_sh_lo, 8 * b, gs);
            quad!(window, cbyte, code_sh_hi, corr_sh_hi, 8 * b + 4, gs);
        }

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON NF4 path (aarch64): vectorised codebook dequant + dot for the
    /// coarse term of a [`FLAG_NF4`] tile. The dispatcher calls this
    /// unconditionally on aarch64, like [`Self::compute_score_neon`].
    ///
    /// Decode strategy: same denibbling as [`Self::compute_score_neon`];
    /// the 16-entry f32 [`NF4_CODEBOOK`] is viewed as a 64-byte table
    /// (aarch64 is little-endian, so byte `k` of level `n` sits at table
    /// index `4n + k`) and one `vqtbl4q_u8` per quad gathers 4 whole levels.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_nf4_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_nf4(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let mask = vdup_n_u8(0x0F);
        let codebook = vld1q_u8_x4(NF4_CODEBOOK.as_ptr() as *const u8);
        // Little-endian byte offsets 0..=3 of an f32, one per index byte.
        let byte_lanes = vdupq_n_u32(0x0302_0100);
        let mut acc = vdupq_n_f32(0.0);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // Gather 4 codebook levels from a quad of nibble indices ($n4:
        // uint32x4, values 0..=15): each u32 lane becomes the four table
        // indices `4n + {0,1,2,3}` (all < 64), one `vqtbl4q_u8` reads the 4
        // f32s whole; then scale and FMA with q.
        macro_rules! quad {
            ($n4:expr, $off:expr, $gs:expr) => {{
                let idx = vaddq_u32(vmulq_n_u32($n4, 0x0404_0404), byte_lanes);
                let level = vreinterpretq_f32_u8(vqtbl4q_u8(codebook, vreinterpretq_u8_u32(idx)));
                let v = vmulq_f32(level, $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 8 groups × 8 bytes = 8 × 16 dims = 128 dims; one scale per group.
        for g in 0..N_GROUPS {
            let base = g * GROUP_DIM;
            let gs = vdupq_n_f32(global * (self.group_scales[g] as f32 * inv255));
            let packed = vld1_u8(latent.add(g * 8)); // 8 bytes
            let lo = vand_u8(packed, mask);
            let hi = vshr_n_u8::<4>(packed);
            // Interleave so nibbles come out in dimension order.
            let d_lo = vzip1_u8(lo, hi); // dims base..base+7
            let d_hi = vzip2_u8(lo, hi); // dims base+8..base+15

            let w_lo = vmovl_u8(d_lo); // u16×8
            quad!(vmovl_u16(vget_low_u16(w_lo)), base, gs);
            quad!(vmovl_u16(vget_high_u16(w_lo)), base + 4, gs);

            let w_hi = vmovl_u8(d_hi);
            quad!(vmovl_u16(vget_low_u16(w_hi)), base + 8, gs);
            quad!(vmovl_u16(vget_high_u16(w_hi)), base + 12, gs);
        }

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON mixed-precision path (aarch64): vectorised dequant + dot for the
    /// coarse term of a [`FLAG_MIXED`] tile. The dispatcher calls this
    /// unconditionally on aarch64, like [`Self::compute_score_neon`].
    ///
    /// Same decode strategy as `Self::compute_score_avx2_mixed`, in
    /// 4-lane quads: the 8-bit head (zero-point 128, scale `gs[0]`) as two
    /// quads, then [`Self::compute_score_neon`]'s denibbling shifted
    /// 8 bytes/8 dims with group scales `gs[1..]`; the dropped tail decodes
    /// to 0 and contributes nothing.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_mixed_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_mixed(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let eight = vdupq_n_f32(8.0);
        let zero_point = vdupq_n_f32(128.0);
        let mask = vdup_n_u8(0x0F);
        let mut acc = vdupq_n_f32(0.0);
        let latent = self.latent_kv.as_ptr();
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS, one signed byte each
        // (zero-point 128, scale gs[0]), as two 4-lane FMAs.
        {
            let hs = vdupq_n_f32(global * (self.group_scales[0] as f32 * inv255));
            let w = vmovl_u8(vld1_u8(latent)); // u16×8
            let n_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
            let n_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_lo, zero_point), hs),
                vld1q_f32(q),
            );
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_hi, zero_point), hs),
                vld1q_f32(q.add(4)),
            );
        }

        // Dequant `(n - 8) * group_scale` for a 4-lane chunk, then FMA with q.
        macro_rules! quad {
            ($n4:expr, $off:expr, $gs:expr) => {{
                let v = vmulq_f32(vsubq_f32($n4, eight), $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // 4-bit body: 7 groups × 8 bytes = 112 dims after the head; one
        // scale byte per 16-dim group (gs[1..]).
        for g in 0..MIXED_LO_GROUPS {
            let base = MIXED_HI_DIMS + g * GROUP_DIM;
            let gs = vdupq_n_f32(global * (self.group_scales[1 + g] as f32 * inv255));
            let packed = vld1_u8(latent.add(MIXED_HI_DIMS + g * 8)); // 8 bytes
            let lo = vand_u8(packed, mask);
            let hi = vshr_n_u8::<4>(packed);
            // Interleave so nibbles come out in dimension order.
            let d_lo = vzip1_u8(lo, hi); // dims base..base+7
            let d_hi = vzip2_u8(lo, hi); // dims base+8..base+15

            let w_lo = vmovl_u8(d_lo); // u16×8
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_lo))), base, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_lo))), base + 4, gs);

            let w_hi = vmovl_u8(d_hi);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w_hi))), base + 8, gs);
            quad!(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w_hi))), base + 12, gs);
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }

    /// NEON MIX3 path (aarch64): vectorised dequant + dot for the coarse
    /// term of a [`FLAG_MIX3`] tile (honours [`FLAG_TQ3_NOCORR`]). The
    /// dispatcher calls this unconditionally on aarch64, like
    /// [`Self::compute_score_neon`].
    ///
    /// Same decode strategy as [`Self::compute_score_neon_tq3`] for the body
    /// — two 4-lane quads per 8-dim block, every byte offset shifted to the
    /// MIX3 planes ([`MIX3_CODES_OFF`] / [`MIX3_CORR_OFF`]) — with the mixed
    /// kernel's two head quads in front.
    ///
    /// Note: compile-checked via cross-compilation to `aarch64-unknown-linux-gnu`;
    /// runtime equivalence is asserted by `neon_mix3_path_matches_scalar` on ARM.
    ///
    /// # Safety
    /// Uses `std::arch::aarch64` NEON intrinsics; sound on any aarch64 CPU.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn compute_score_neon_mix3(
        &self,
        q_coarse: &[f32; D_C],
        q_sign: &[u64; RESIDUAL_WORDS],
    ) -> f32 {
        use std::arch::aarch64::*;

        // Negative counts = right shifts: code fields at bits 3l, correction
        // bits at bit l, for lanes l of the low/high quad of an 8-dim block.
        const CODE_SH_LO: [i32; 4] = [0, -3, -6, -9];
        const CODE_SH_HI: [i32; 4] = [-12, -15, -18, -21];
        const CORR_SH_LO: [i32; 4] = [0, -1, -2, -3];
        const CORR_SH_HI: [i32; 4] = [-4, -5, -6, -7];

        let global = self.scale;
        let inv255 = 1.0f32 / 255.0;
        let code_sh_lo = vld1q_s32(CODE_SH_LO.as_ptr());
        let code_sh_hi = vld1q_s32(CODE_SH_HI.as_ptr());
        let corr_sh_lo = vld1q_s32(CORR_SH_LO.as_ptr());
        let corr_sh_hi = vld1q_s32(CORR_SH_HI.as_ptr());
        let seven = vdupq_n_u32(7);
        let one = vdupq_n_u32(1);
        let zero_point = vdupq_n_f32(128.0);
        let nocorr = self.is_tq3_nocorr();
        // level = code − 3.5 ± 0.25 = code − 3.75 + corr·0.5 (exact in f32).
        let bias = vdupq_n_f32(if nocorr {
            TQ3_HALF_RANGE
        } else {
            TQ3_HALF_RANGE + TQ3_CORRECTION
        });
        let corr_step = vdupq_n_f32(2.0 * TQ3_CORRECTION);
        let mut acc = vdupq_n_f32(0.0);
        let lk = &self.latent_kv;
        let q = q_coarse.as_ptr();

        // 8-bit head: dims 0..MIXED_HI_DIMS (zero-point 128, scale gs[0]),
        // exactly the mixed kernel's two quads.
        {
            let hs = vdupq_n_f32(global * (self.group_scales[0] as f32 * inv255));
            let w = vmovl_u8(vld1_u8(lk.as_ptr())); // u16×8
            let n_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
            let n_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_lo, zero_point), hs),
                vld1q_f32(q),
            );
            acc = vfmaq_f32(
                acc,
                vmulq_f32(vsubq_f32(n_hi, zero_point), hs),
                vld1q_f32(q.add(4)),
            );
        }

        // Decode 4 dims from the block's window/correction byte, FMA with q.
        macro_rules! quad {
            ($window:expr, $cbyte:expr, $code_sh:expr, $corr_sh:expr, $off:expr, $gs:expr) => {{
                let codes = vandq_u32(vshlq_u32(vdupq_n_u32($window), $code_sh), seven);
                let mut level = vsubq_f32(vcvtq_f32_u32(codes), bias);
                if !nocorr {
                    let corr = vandq_u32(vshlq_u32(vdupq_n_u32($cbyte), $corr_sh), one);
                    level = vfmaq_f32(level, vcvtq_f32_u32(corr), corr_step);
                }
                let v = vmulq_f32(level, $gs);
                acc = vfmaq_f32(acc, v, vld1q_f32(q.add($off)));
            }};
        }

        // TQ3 body: 14 blocks × 8 dims (3 code bytes + 1 correction byte
        // each) = 112 dims after the head; one scale per 16-dim group = two
        // consecutive blocks (gs[1..]).
        for b in 0..MIXED_LO_DIMS / 8 {
            let gs = vdupq_n_f32(global * (self.group_scales[1 + b / 2] as f32 * inv255));
            // 4-byte little-endian window at MIX3_CODES_OFF + 3b holding all
            // 8 code fields of this block. The last block (b = 13) reads
            // bytes 47..=50; byte 50 is the first correction byte
            // (MIX3_CORR_OFF) — in-bounds of `latent_kv`, and its bits sit
            // at window positions >= 24 > 3l + 2 for every lane l <= 7, so
            // the `& 7` mask after the per-lane shift keeps them out of
            // every decoded code.
            let o = MIX3_CODES_OFF + 3 * b;
            let window = u32::from_le_bytes([lk[o], lk[o + 1], lk[o + 2], lk[o + 3]]);
            let cbyte = u32::from(lk[MIX3_CORR_OFF + b]);
            quad!(
                window,
                cbyte,
                code_sh_lo,
                corr_sh_lo,
                MIXED_HI_DIMS + 8 * b,
                gs
            );
            quad!(
                window,
                cbyte,
                code_sh_hi,
                corr_sh_hi,
                MIXED_HI_DIMS + 8 * b + 4,
                gs
            );
        }
        // Dims MIXED_DIMS..D_C are dropped (decode to 0): no contribution.

        let coarse = vaddvq_f32(acc);
        if self.is_warm() {
            return coarse;
        }
        coarse + self.residual_term(q_sign)
    }
}

/// Hamming distance over the 256-bit residual: `Σ popcount(aᵢ ⊕ bᵢ)`, the hot
/// inner term of eq. (2.3) (`d_s − 2·Hamming` is the signed ±1 dot product).
///
/// **Branchless**: a fixed 4-word reduction with no data-dependent control flow
/// (important for in-order issue / tight pipelines like ARM Neoverse). On
/// x86-64 CPUs advertising AVX-512 `VPOPCNTDQ`+`VL` it folds the whole 256-bit
/// residual into a single vector `vpopcntq`; otherwise it falls back to
/// `u64::count_ones`, which already lowers to `POPCNT` (x86) / `CNT` (AArch64
/// NEON). The one-time feature-detection branch is predicted and cached.
#[inline]
pub fn hamming_distance(a: &[u64; RESIDUAL_WORDS], b: &[u64; RESIDUAL_WORDS]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512vl") {
            // SAFETY: both required target features were just detected at runtime.
            return unsafe { hamming_vpopcntdq(a, b) };
        }
    }
    hamming_scalar(a, b)
}

/// Portable branchless reference: 4× `count_ones` (→ `POPCNT`/`CNT`).
#[inline]
pub(super) fn hamming_scalar(a: &[u64; RESIDUAL_WORDS], b: &[u64; RESIDUAL_WORDS]) -> u32 {
    let mut h = 0u32;
    for w in 0..RESIDUAL_WORDS {
        h += (a[w] ^ b[w]).count_ones();
    }
    h
}

/// AVX-512 VPOPCNTDQ path: XOR the two 256-bit residuals and popcount all four
/// 64-bit lanes in a single `vpopcntq`, then reduce. Equivalent to
/// [`hamming_scalar`] (asserted by `vpopcntdq_hamming_matches_scalar` on capable
/// CPUs; compile-checked elsewhere).
///
/// # Safety
/// Requires the `avx512vpopcntdq` and `avx512vl` target features; the only
/// caller ([`hamming_distance`]) gates this behind `is_x86_feature_detected!`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vpopcntdq,avx512vl")]
pub(super) unsafe fn hamming_vpopcntdq(
    a: &[u64; RESIDUAL_WORDS],
    b: &[u64; RESIDUAL_WORDS],
) -> u32 {
    use core::arch::x86_64::*;
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let pc = _mm256_popcnt_epi64(_mm256_xor_si256(va, vb)); // per-lane popcount
    let mut lanes = [0u64; RESIDUAL_WORDS];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, pc);
    (lanes[0] + lanes[1] + lanes[2] + lanes[3]) as u32
}
