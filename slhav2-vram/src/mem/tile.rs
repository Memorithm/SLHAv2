use scirust::SciRustSlhaTile;

use crate::codec;

/// Canonical 128-byte serialized SLHAv2 tile.
pub struct SerializedTile(pub [u8; codec::TILE_BYTES]);

impl SerializedTile {
    pub fn new() -> Self {
        Self([0u8; codec::TILE_BYTES])
    }

    pub fn zeroed() -> Self {
        Self([0u8; codec::TILE_BYTES])
    }

    /// Copy up to 128 bytes and zero-pad the remainder.
    ///
    /// Use [`Self::try_from_exact_bytes`] for untrusted/external serialized
    /// data where accepting a truncated tile would hide an input error.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut inner = [0u8; codec::TILE_BYTES];
        let len = bytes.len().min(codec::TILE_BYTES);
        inner[..len].copy_from_slice(&bytes[..len]);
        Self(inner)
    }

    /// Parse exactly one serialized tile.
    pub fn try_from_exact_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        let inner: [u8; codec::TILE_BYTES] = bytes
            .try_into()
            .map_err(|_| "serialized tile must be exactly 128 bytes")?;
        Ok(Self(inner))
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
        for (i, chunk) in self.raw_residual().chunks_exact(8).enumerate() {
            out[i] = u64::from_le_bytes(chunk.try_into().expect("residual chunk is 8 bytes"));
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
        self.read_f32(codec::SCALE_OFFSET)
    }

    pub fn set_scale(&mut self, val: f32) {
        self.write_f32(codec::SCALE_OFFSET, val);
    }

    pub fn dynamic_lambda(&self) -> f32 {
        self.read_f32(codec::DYNAMIC_LAMBDA_OFFSET)
    }

    pub fn set_dynamic_lambda(&mut self, val: f32) {
        self.write_f32(codec::DYNAMIC_LAMBDA_OFFSET, val);
    }

    pub fn residual_sigma(&self) -> f32 {
        self.read_f32(codec::RESIDUAL_SIGMA_OFFSET)
    }

    pub fn set_residual_sigma(&mut self, val: f32) {
        self.write_f32(codec::RESIDUAL_SIGMA_OFFSET, val);
    }

    pub fn token_id(&self) -> u32 {
        self.read_u32(codec::TOKEN_ID_OFFSET)
    }

    pub fn set_token_id(&mut self, val: u32) {
        self.write_u32(codec::TOKEN_ID_OFFSET, val);
    }

    pub fn position(&self) -> u32 {
        self.read_u32(codec::POSITION_OFFSET)
    }

    pub fn set_position(&mut self, val: u32) {
        self.write_u32(codec::POSITION_OFFSET, val);
    }

    pub fn head_id(&self) -> u16 {
        u16::from_le_bytes(
            self.0[codec::HEAD_ID_OFFSET..codec::HEAD_ID_OFFSET + 2]
                .try_into()
                .expect("head_id field in bounds"),
        )
    }

    pub fn set_head_id(&mut self, val: u16) {
        self.0[codec::HEAD_ID_OFFSET..codec::HEAD_ID_OFFSET + 2]
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

    /// Score this tile against a query, validating codec flags first.
    pub fn try_score(&self, q_coarse: &[f32], q_sign: &[u64]) -> Result<f32, codec::CodecError> {
        codec::validate_codec(self.flags())?;
        Ok(self.score_unchecked(q_coarse, q_sign))
    }

    /// Score this tile against a query.
    ///
    /// # Panics
    /// Panics on invalid codec flags or query dimensions.
    pub fn score(&self, q_coarse: &[f32], q_sign: &[u64]) -> f32 {
        codec::validate_codec(self.flags()).unwrap_or_else(|e| panic!("cannot score tile: {e}"));
        self.score_unchecked(q_coarse, q_sign)
    }

    fn score_unchecked(&self, q_coarse: &[f32], q_sign: &[u64]) -> f32 {
        let q_coarse: &[f32; codec::D_C] = q_coarse
            .try_into()
            .expect("q_coarse slice must be exactly D_C elements");
        let q_sign: &[u64; codec::RESIDUAL_WORDS] = q_sign
            .try_into()
            .expect("q_sign slice must be exactly RESIDUAL_WORDS elements");
        self.to_slha_tile().compute_score(q_coarse, q_sign)
    }

    /// Convert without losing any field present in the 128-byte wire layout.
    pub fn to_slha_tile(&self) -> SciRustSlhaTile {
        let mut latent_kv = [0u8; codec::LATENT_BYTES];
        latent_kv.copy_from_slice(self.latent());
        let mut group_scales = [0u8; codec::N_GROUP_SCALES];
        group_scales.copy_from_slice(self.group_scales());
        SciRustSlhaTile {
            latent_kv,
            residual_bitmap: self.residual(),
            scale: self.scale(),
            dynamic_lambda: self.dynamic_lambda(),
            residual_sigma: self.residual_sigma(),
            token_id: self.token_id(),
            position: self.position(),
            head_id: self.head_id(),
            flags: self.flags(),
            group_scales,
        }
    }

    fn read_f32(&self, offset: usize) -> f32 {
        codec::read_f32_le(&self.0, offset).expect("f32 field in bounds")
    }

    fn write_f32(&mut self, offset: usize, val: f32) {
        self.0[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn read_u32(&self, offset: usize) -> u32 {
        u32::from_le_bytes(
            self.0[offset..offset + 4]
                .try_into()
                .expect("u32 field in bounds"),
        )
    }

    fn write_u32(&mut self, offset: usize, val: u32) {
        self.0[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
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
    fn exact_parser_rejects_truncation() {
        assert!(SerializedTile::try_from_exact_bytes(&[0u8; 127]).is_err());
        assert!(SerializedTile::try_from_exact_bytes(&[0u8; 129]).is_err());
        assert!(SerializedTile::try_from_exact_bytes(&[0u8; 128]).is_ok());
    }

    #[test]
    fn scalar_fields_roundtrip() {
        let mut tile = SerializedTile::new();
        tile.set_scale(2.5);
        tile.set_dynamic_lambda(0.05);
        tile.set_residual_sigma(0.75);
        tile.set_token_id(1234);
        tile.set_position(5678);
        tile.set_head_id(42);
        tile.set_flags(codec::FLAG_NF4);
        tile.set_group_scales(&[1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(tile.scale(), 2.5);
        assert!((tile.dynamic_lambda() - 0.05).abs() < 1e-6);
        assert_eq!(tile.residual_sigma(), 0.75);
        assert_eq!(tile.token_id(), 1234);
        assert_eq!(tile.position(), 5678);
        assert_eq!(tile.head_id(), 42);
        assert!(tile.is_nf4());
        assert_eq!(tile.group_scales(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn residual_roundtrip() {
        let mut tile = SerializedTile::new();
        let res = [0xDEAD_BEEF_CAFE_FACEu64, 1, 2, 3];
        tile.set_residual(&res);
        assert_eq!(tile.residual(), res);
    }

    #[test]
    fn to_slha_tile_preserves_all_metadata() {
        let mut tile = SerializedTile::new();
        tile.set_scale(2.5);
        tile.set_dynamic_lambda(0.05);
        tile.set_residual_sigma(0.75);
        tile.set_token_id(1234);
        tile.set_position(5678);
        tile.set_head_id(42);
        tile.set_flags(codec::FLAG_NF4);
        tile.set_group_scales(&[10, 20, 30, 40, 50, 60, 70, 80]);
        tile.latent_mut()[0] = 0x12;
        tile.set_residual(&[0xAAAA_BBBB_CCCC_DDDDu64, 1, 2, 3]);

        let st = tile.to_slha_tile();
        assert_eq!(st.scale, 2.5);
        assert!((st.dynamic_lambda - 0.05).abs() < 1e-6);
        assert_eq!(st.residual_sigma, 0.75);
        assert_eq!(st.token_id, 1234);
        assert_eq!(st.position, 5678);
        assert_eq!(st.head_id, 42);
        assert!(st.is_nf4());
        assert_eq!(st.latent_kv[0], 0x12);
        assert_eq!(st.residual_bitmap[0], 0xAAAA_BBBB_CCCC_DDDDu64);
        assert_eq!(st.group_scales[7], 80);
    }

    #[test]
    fn serialized_size_is_stable() {
        assert_eq!(core::mem::size_of::<SerializedTile>(), codec::TILE_BYTES);
    }
}
