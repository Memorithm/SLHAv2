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

    pub fn latent_kv(&self) -> &[u8] {
        &self.0[..codec::LATENT_KV_WORDS]
    }

    pub fn residual(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.0[codec::RESIDUAL_OFFSET..].as_ptr() as *const u64,
                codec::RESIDUAL_WORDS,
            )
        }
    }

    pub fn scale(&self) -> f32 {
        f32::from_le_bytes(
            self.0[codec::SCALE_OFFSET..codec::SCALE_OFFSET + 4]
                .try_into()
                .unwrap(),
        )
    }

    pub fn set_scale(&mut self, val: f32) {
        self.0[codec::SCALE_OFFSET..codec::SCALE_OFFSET + 4]
            .copy_from_slice(&val.to_le_bytes());
    }

    pub fn dynamic_lambda(&self) -> f32 {
        f32::from_le_bytes(
            self.0[codec::DYNAMIC_LAMBDA_OFFSET..codec::DYNAMIC_LAMBDA_OFFSET + 4]
                .try_into()
                .unwrap(),
        )
    }

    pub fn set_dynamic_lambda(&mut self, val: f32) {
        self.0[codec::DYNAMIC_LAMBDA_OFFSET..codec::DYNAMIC_LAMBDA_OFFSET + 4]
            .copy_from_slice(&val.to_le_bytes());
    }

    pub fn flags(&self) -> u16 {
        u16::from_le_bytes(
            self.0[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2]
                .try_into()
                .unwrap(),
        )
    }

    pub fn set_flags(&mut self, val: u16) {
        self.0[codec::FLAGS_OFFSET..codec::FLAGS_OFFSET + 2]
            .copy_from_slice(&val.to_le_bytes());
    }

    pub fn is_warm(&self) -> bool {
        self.flags() & codec::FLAG_WARM != 0
    }

    pub fn group_scales(&self) -> &[u8] {
        &self.0[codec::GROUP_SCALES_OFFSET..codec::GROUP_SCALES_OFFSET + 8]
    }

    pub fn set_group_scales(&mut self, scales: &[u8]) {
        let len = scales.len().min(8);
        self.0[codec::GROUP_SCALES_OFFSET..codec::GROUP_SCALES_OFFSET + len]
            .copy_from_slice(&scales[..len]);
    }

    pub fn score(&self, q_coarse: &[f32], q_sign: &[u64]) -> f32 {
        if self.is_warm() {
            codec::score_warm(
                q_coarse,
                &self.0[..codec::LATENT_KV_WORDS],
                self.scale(),
                self.group_scales(),
                self.flags(),
            )
        } else {
            codec::score_hot(
                q_coarse,
                q_sign,
                &self.0[..codec::LATENT_KV_WORDS],
                self.residual(),
                self.scale(),
                self.dynamic_lambda(),
                self.group_scales(),
                self.flags(),
            )
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
}
