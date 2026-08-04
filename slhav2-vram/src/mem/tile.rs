use scirust::SciRustSlhaTile;

use crate::codec;

pub struct SerializedTile(pub [u8; codec::TILE_BYTES]);

impl SerializedTile {
    pub fn new() -> Self {
        Self([0u8; codec::TILE_BYTES])
    }

    pub fn zeroed() -> Self {
        Self([0u8; codec::TILE_BYTES])
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut inner = [0u8; codec::TILE_BYTES];
        let len = bytes.len().min(codec::TILE_BYTES);
        inner[..len].copy_from_slice(&bytes[..len]);
        Self(inner)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    pub fn latent(&self) -> &[u8] {
        &self.0[..codec::LATENT_BYTES]
    }

    pub fn latent_mut(&mut self) -> &mut [u8] {
        &mut self.0[..codec::LATENT_BYTES]
    }

    pub fn raw_residual(&self) -> &[u8] {
        &self.0[codec::RESIDUAL_OFFSET..codec::RESIDUAL_OFFSET + codec::RESIDUAL_WORDS * 8]
    }

    pub fn residual(&self) -> [u64; codec::RESIDUAL_WORDS] {
        let mut out = [0u64; codec::RESIDUAL_WORDS];
        let raw = self.raw_residual();
        for (i, chunk) in raw.chunks_exact(8).enumerate() {
            let arr: [u8; 8] = chunk.try_into().expect("residual chunk is 8 bytes");
            out[i] = u64::from_le_bytes(arr);
        }
        out
    }

    pub fn set_residual(&mut self, val: &[u64; codec::RESIDUAL_WORDS]) {
        for (i, &v) in val.iter().enumerate() {
            let off = codec::RESIDUAL_OFFSET + i * 8;
            self.0[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn scale(&self) -> f32 {
        codec::read_f32_le(&self.0, codec::SCALE_OFFSET).expect("scale field in bounds")
    }

    pub fn set_scale(&mut self, val: f32) {
        self.0[codec::SCALE_OFFSET..codec::SCALE_OFFSET + 4].copy_from_slice(&val.to_le_bytes());
    }

    pub fn dynamic_lambda(&self) -> f32 {
        codec::read_f32_le(&self.0, codec::DYNAMIC_LAMBDA_OFFSET)
            .expect("dynamic_lambda field in bounds")
    }

    pub fn set_dynamic_lambda(&mut self, val: f32) {
        self.0[codec::DYNAMIC_LAMBDA_OFFSET..codec::DYNAMIC_LAMBDA_OFFSET + 4]
            .copy_from_slice(&val.to_le_bytes());
    }

    pub fn flags(&self) -> u16 {
        codec::read_u16_le(&self.0, codec::FLAGS_OFFSET).expect("flags field in bounds")
    }

    pub fn set_flags(&mut self, val: u16) {
        self.0[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2].copy_from_slice(&val.to_le_bytes());
    }

    pub fn is_warm(&self) -> bool {
        self.flags() & codec::FLAG_WARM != 0
    }

    pub fn is_nf4(&self) -> bool {
        self.flags() & codec::FLAG_NF4 != 0
    }

    pub fn group_scales(&self) -> &[u8] {
        &self.0[codec::GROUP_SCALES_OFFSET..codec::GROUP_SCALES_OFFSET + codec::N_GROUP_SCALES]
    }

    pub fn set_group_scales(&mut self, scales: &[u8]) {
        let len = scales.len().min(codec::N_GROUP_SCALES);
        self.0[codec::GROUP_SCALES_OFFSET..codec::GROUP_SCALES_OFFSET + len]
            .copy_from_slice(&scales[..len]);
    }

    /// Score this tile against a query, validating the codec flags first.
    ///
    /// Returns `Err(CodecError)` for an unsupported flag combination (see
    /// [`codec::validate_codec`]) instead of silently decoding as INT4.
    pub fn try_score(&self, q_coarse: &[f32], q_sign: &[u64]) -> Result<f32, codec::CodecError> {
        codec::validate_codec(self.flags())?;
        Ok(self.score_unchecked(q_coarse, q_sign))
    }

    /// Score this tile against a query.
    ///
    /// # Panics
    ///
    /// Panics with a descriptive message if the tile's flags select an
    /// unsupported codec combination. Prefer [`Self::try_score`] when handling
    /// untrusted tiles.
    pub fn score(&self, q_coarse: &[f32], q_sign: &[u64]) -> f32 {
        codec::validate_codec(self.flags()).unwrap_or_else(|e| panic!("cannot score tile: {e}"));
        self.score_unchecked(q_coarse, q_sign)
    }

    fn score_unchecked(&self, q_coarse: &[f32], q_sign: &[u64]) -> f32 {
        let latent = self.latent();
        let scale = self.scale();
        let gscales = self.group_scales();
        let flags = self.flags();
        if self.is_warm() {
            codec::score_warm(q_coarse, latent, scale, gscales, flags)
        } else {
            let residual = self.residual();
            codec::score_hot(
                q_coarse,
                q_sign,
                latent,
                &residual,
                scale,
                self.dynamic_lambda(),
                gscales,
                flags,
            )
        }
    }

    pub fn to_slha_tile(&self) -> SciRustSlhaTile {
        let residual = self.residual();
        SciRustSlhaTile {
            latent_kv: {
                let mut lk = [0u8; codec::LATENT_BYTES];
                lk.copy_from_slice(self.latent());
                lk
            },
            residual_bitmap: residual,
            scale: self.scale(),
            dynamic_lambda: self.dynamic_lambda(),
            residual_sigma: 0.0,
            token_id: 0,
            position: 0,
            head_id: 0,
            flags: self.flags(),
            group_scales: {
                let mut gs = [0u8; codec::N_GROUP_SCALES];
                gs.copy_from_slice(self.group_scales());
                gs
            },
        }
    }
}

impl Default for SerializedTile {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SerializedTile {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_roundtrip() {
        let mut tile = SerializedTile::new();
        tile.set_scale(1.5);
        tile.set_dynamic_lambda(0.1);
        tile.set_flags(0);

        assert!((tile.scale() - 1.5).abs() < 1e-6);
        assert!((tile.dynamic_lambda() - 0.1).abs() < 1e-6);
        assert_eq!(tile.flags(), 0);
        assert!(!tile.is_warm());

        tile.set_flags(codec::FLAG_WARM);
        assert!(tile.is_warm());
    }

    #[test]
    fn test_tile_size() {
        assert_eq!(std::mem::size_of::<SerializedTile>(), codec::TILE_BYTES);
    }

    #[test]
    fn test_residual_roundtrip() {
        let mut tile = SerializedTile::new();
        let res = [0xDEAD_BEEF_CAFE_FACEu64, 1, 2, 3];
        tile.set_residual(&res);
        let got = tile.residual();
        assert_eq!(res, got);
    }

    #[test]
    fn test_latent_access() {
        let mut tile = SerializedTile::new();
        tile.latent_mut()[0] = 0xAB;
        tile.latent_mut()[codec::LATENT_BYTES - 1] = 0xCD;
        assert_eq!(tile.latent()[0], 0xAB);
        assert_eq!(tile.latent()[codec::LATENT_BYTES - 1], 0xCD);
    }

    #[test]
    fn test_to_slha_tile_roundtrip() {
        let mut tile = SerializedTile::new();
        tile.set_scale(2.5);
        tile.set_dynamic_lambda(0.05);
        tile.set_flags(codec::FLAG_NF4);
        tile.set_group_scales(&[1, 2, 3, 4, 5, 6, 7, 8]);
        tile.latent_mut()[0] = 0x12;
        tile.latent_mut()[31] = 0x34;
        tile.set_residual(&[0xAAAA_BBBB_CCCC_DDDDu64, 0, 0, 0]);

        let st = tile.to_slha_tile();
        assert_eq!(st.scale, 2.5);
        assert!((st.dynamic_lambda - 0.05).abs() < 1e-6);
        assert!(st.is_nf4());
        assert_eq!(st.latent_kv[0], 0x12);
        assert_eq!(st.latent_kv[31], 0x34);
        assert_eq!(st.residual_bitmap[0], 0xAAAA_BBBB_CCCC_DDDDu64);
        assert_eq!(st.group_scales[0], 1);
        assert_eq!(st.group_scales[7], 8);
    }
}
