use crate::codec;
use crate::mem::tile::SerializedTile;
use crate::traits::DeviceEngine;

/// Inputs for CPU reference scoring.
pub struct ScoringInput<'a, E: DeviceEngine> {
    pub engine: &'a E,
    pub tiles: &'a [SerializedTile],
    pub q_coarse: &'a [f32],
    pub q_sign: &'a [u64],
    pub scores: &'a mut [f32],
}

/// Checked CPU reference scoring.
pub fn try_score_tiles_cpu<E: DeviceEngine>(input: ScoringInput<'_, E>) -> Result<(), String> {
    if input.q_coarse.len() != codec::D_C {
        return Err(format!(
            "q_coarse must contain exactly {} f32 values, got {}",
            codec::D_C,
            input.q_coarse.len()
        ));
    }
    if input.q_sign.len() != codec::RESIDUAL_WORDS {
        return Err(format!(
            "q_sign must contain exactly {} u64 values, got {}",
            codec::RESIDUAL_WORDS,
            input.q_sign.len()
        ));
    }
    if input.scores.len() < input.tiles.len() {
        return Err(format!(
            "scores has {} entries but {} tiles require scores",
            input.scores.len(),
            input.tiles.len()
        ));
    }
    for (index, tile) in input.tiles.iter().enumerate() {
        input.scores[index] = tile
            .try_score(input.q_coarse, input.q_sign)
            .map_err(|error| format!("tile {index}: {error}"))?;
    }
    Ok(())
}

/// Compatibility wrapper for CPU scoring.
///
/// Invalid input fails closed by zeroing the output instead of indexing past a
/// caller-provided buffer. New code should use [`try_score_tiles_cpu`].
pub fn score_tiles_cpu<E: DeviceEngine>(input: ScoringInput<'_, E>) {
    input.scores.fill(0.0);
    let _ = try_score_tiles_cpu(input);
}

/// Copy trusted serialized tiles to a generic device allocation.
///
/// CUDA production paths should prefer `GpuScoringPipeline`, which validates
/// codec flags before device mutation when the CUDA PTX path is available.
pub fn copy_tiles_to_gpu<E: DeviceEngine>(
    engine: &E,
    tiles: &[SerializedTile],
    dst_alloc: &mut E::Alloc,
    offset: usize,
) -> Result<(), E::Error> {
    let total = tiles
        .len()
        .checked_mul(codec::TILE_BYTES)
        .expect("copy_tiles_to_gpu: byte count overflow");
    let mut buffer = vec![0u8; total];
    for (index, tile) in tiles.iter().enumerate() {
        let start = index * codec::TILE_BYTES;
        buffer[start..start + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    engine.copy_to_device(&buffer, dst_alloc, offset)
}

/// Copy little-endian f32 scores from a generic device allocation.
pub fn copy_scores_from_gpu<E: DeviceEngine>(
    engine: &E,
    src_alloc: &E::Alloc,
    offset: usize,
    num_scores: usize,
) -> Result<Vec<f32>, E::Error> {
    let total = num_scores
        .checked_mul(core::mem::size_of::<f32>())
        .expect("copy_scores_from_gpu: byte count overflow");
    let mut buffer = vec![0u8; total];
    engine.copy_to_host(src_alloc, offset, &mut buffer)?;
    Ok(buffer
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// Persistent validated CUDA scoring pipeline.
///
/// Serialized 128-byte tiles remain resident in one device allocation. New
/// tiles can be appended with [`Self::upload`], while existing slots can be
/// replaced with [`Self::update_slot`] when an elastic transition changes a
/// tile representation/flags. Every tile is codec-validated before host mirror
/// or device state is modified.
#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
pub struct GpuScoringPipeline {
    engine: crate::backends::cuda::CudaEngine,
    kernel: crate::backends::cuda::CudaFunction,
    stream: crate::backends::cuda::CudaStream,
    tiles_dev: crate::backends::cuda::CudaAllocation,
    scores_dev: crate::backends::cuda::CudaAllocation,
    q_coarse_dev: crate::backends::cuda::CudaAllocation,
    q_sign_dev: crate::backends::cuda::CudaAllocation,
    host_tiles: Vec<u8>,
    resident: usize,
    staging: Vec<u8>,
    score_buf: Vec<u8>,
    q_coarse_buf: Vec<u8>,
    q_sign_buf: Vec<u8>,
}

#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
impl GpuScoringPipeline {
    /// Create a pipeline backed by `capacity_bytes` of device tile storage.
    pub fn new(
        engine: &crate::backends::cuda::CudaEngine,
        kernel: &crate::backends::cuda::CudaFunction,
        capacity_bytes: usize,
    ) -> Result<Self, crate::backends::cuda::CudaError> {
        let max_tiles = capacity_bytes / codec::TILE_BYTES;
        if max_tiles == 0 {
            return Err(crate::backends::cuda::CudaError(
                "GpuScoringPipeline: capacity too small for one tile".into(),
            ));
        }
        let usable_tile_bytes = max_tiles
            .checked_mul(codec::TILE_BYTES)
            .ok_or_else(|| crate::backends::cuda::CudaError("tile capacity overflow".into()))?;
        let tiles_dev = engine.allocate(usable_tile_bytes)?;
        let scores_dev = engine.allocate(max_tiles * core::mem::size_of::<f32>())?;
        let q_coarse_dev = engine.allocate(codec::D_C * core::mem::size_of::<f32>())?;
        let q_sign_dev = engine.allocate(codec::RESIDUAL_WORDS * core::mem::size_of::<u64>())?;
        let stream = crate::backends::cuda::CudaStream::new(engine)?;
        Ok(Self {
            engine: engine.clone(),
            kernel: kernel.clone(),
            stream,
            tiles_dev,
            scores_dev,
            q_coarse_dev,
            q_sign_dev,
            host_tiles: vec![0u8; usable_tile_bytes],
            resident: 0,
            staging: vec![0u8; codec::TILE_BYTES],
            score_buf: Vec::with_capacity(max_tiles * core::mem::size_of::<f32>()),
            q_coarse_buf: Vec::with_capacity(codec::D_C * core::mem::size_of::<f32>()),
            q_sign_buf: Vec::with_capacity(codec::RESIDUAL_WORDS * core::mem::size_of::<u64>()),
        })
    }

    /// Append validated tiles and return the first assigned slot.
    ///
    /// Validation happens for the complete batch before the first device copy,
    /// preventing a partially appended batch when a later tile has invalid
    /// codec flags.
    pub fn upload(
        &mut self,
        tiles: &[SerializedTile],
    ) -> Result<usize, crate::backends::cuda::CudaError> {
        if tiles.is_empty() {
            return Ok(self.resident);
        }
        self.validate_tiles(tiles)?;
        let start = self.resident;
        let needed = start.checked_add(tiles.len()).ok_or_else(|| {
            crate::backends::cuda::CudaError("resident tile count overflow".into())
        })?;
        if needed > self.capacity_tiles() {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: arena holds {} tiles, need {needed}",
                self.capacity_tiles()
            )));
        }

        for (offset, tile) in tiles.iter().enumerate() {
            self.write_slot(start + offset, tile)?;
        }
        self.resident = needed;
        Ok(start)
    }

    /// Replace an existing resident slot after validating the tile.
    pub fn update_slot(
        &mut self,
        slot: usize,
        tile: &SerializedTile,
    ) -> Result<(), crate::backends::cuda::CudaError> {
        if slot >= self.resident {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: slot {slot} is not resident (resident={})",
                self.resident
            )));
        }
        self.validate_tile(slot, tile)?;
        self.write_slot(slot, tile)
    }

    /// Score all resident tiles. Query dimensions must be exact.
    pub fn score_into(
        &mut self,
        q_coarse: &[f32],
        q_sign: &[u64],
        scores: &mut [f32],
    ) -> Result<(), crate::backends::cuda::CudaError> {
        if q_coarse.len() != codec::D_C {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: q_coarse must be exactly {} f32 values, got {}",
                codec::D_C,
                q_coarse.len()
            )));
        }
        if q_sign.len() != codec::RESIDUAL_WORDS {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: q_sign must be exactly {} u64 values, got {}",
                codec::RESIDUAL_WORDS,
                q_sign.len()
            )));
        }
        if scores.len() < self.resident {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: scores buffer has {} entries, need {}",
                scores.len(),
                self.resident
            )));
        }
        if self.resident == 0 {
            return Ok(());
        }

        self.q_coarse_buf.clear();
        for &value in q_coarse {
            self.q_coarse_buf.extend_from_slice(&value.to_le_bytes());
        }
        self.q_sign_buf.clear();
        for &value in q_sign {
            self.q_sign_buf.extend_from_slice(&value.to_le_bytes());
        }
        self.engine
            .copy_to_device_at(&self.q_coarse_buf, &mut self.q_coarse_dev, 0)?;
        self.engine
            .copy_to_device_at(&self.q_sign_buf, &mut self.q_sign_dev, 0)?;

        self.engine.score_tiles_on_stream(
            &self.q_coarse_dev,
            &self.q_sign_dev,
            &self.tiles_dev,
            &self.scores_dev,
            self.resident as i32,
            &self.kernel,
            &self.stream,
        )?;

        let nbytes = self
            .resident
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| crate::backends::cuda::CudaError("score byte count overflow".into()))?;
        self.score_buf.resize(nbytes, 0);
        // The scoring kernel executes asynchronously on `self.stream`.
        // `score_buf` is ordinary Rust-owned pageable memory, so CUDA must
        // not retain its pointer beyond this mutable borrow. Complete the
        // stream first, then perform the synchronous device-to-host copy.
        self.stream.synchronize()?;
        self.engine
            .copy_to_host(&self.scores_dev, 0, &mut self.score_buf)?;

        for (index, chunk) in self.score_buf.as_chunks::<4>().0.iter().enumerate() {
            scores[index] = f32::from_le_bytes(*chunk);
        }
        Ok(())
    }

    /// Number of active resident slots.
    pub fn resident(&self) -> usize {
        self.resident
    }

    /// Maximum number of serialized tiles in the device arena.
    pub fn capacity_tiles(&self) -> usize {
        self.host_tiles.len() / codec::TILE_BYTES
    }

    fn validate_tiles(
        &self,
        tiles: &[SerializedTile],
    ) -> Result<(), crate::backends::cuda::CudaError> {
        for (offset, tile) in tiles.iter().enumerate() {
            self.validate_tile(self.resident + offset, tile)?;
        }
        Ok(())
    }

    fn validate_tile(
        &self,
        slot: usize,
        tile: &SerializedTile,
    ) -> Result<(), crate::backends::cuda::CudaError> {
        codec::validate_codec(tile.flags()).map_err(|error| {
            crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: invalid codec flags for slot {slot}: {error}"
            ))
        })
    }

    fn write_slot(
        &mut self,
        slot: usize,
        tile: &SerializedTile,
    ) -> Result<(), crate::backends::cuda::CudaError> {
        let offset = slot
            .checked_mul(codec::TILE_BYTES)
            .ok_or_else(|| crate::backends::cuda::CudaError("tile offset overflow".into()))?;
        let end = offset
            .checked_add(codec::TILE_BYTES)
            .ok_or_else(|| crate::backends::cuda::CudaError("tile end overflow".into()))?;
        if end > self.host_tiles.len() {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: slot {slot} exceeds arena capacity {}",
                self.capacity_tiles()
            )));
        }
        if self.host_tiles[offset..end] == tile.0 {
            return Ok(());
        }
        self.staging.copy_from_slice(&tile.0);
        self.engine
            .copy_to_device_at(&self.staging, &mut self.tiles_dev, offset)?;
        self.host_tiles[offset..end].copy_from_slice(&tile.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::cpu::CpuEngine;

    fn valid_tile() -> SerializedTile {
        let mut tile = SerializedTile::zeroed();
        tile.set_scale(1.0);
        tile.set_group_scales(&[255; codec::N_GROUP_SCALES]);
        tile
    }

    #[test]
    fn checked_cpu_scoring_rejects_short_output() {
        let engine = CpuEngine::new();
        let tiles = [valid_tile(), valid_tile()];
        let query = [0.0f32; codec::D_C];
        let signs = [0u64; codec::RESIDUAL_WORDS];
        let mut scores = [1.0f32; 1];
        let result = try_score_tiles_cpu(ScoringInput {
            engine: &engine,
            tiles: &tiles,
            q_coarse: &query,
            q_sign: &signs,
            scores: &mut scores,
        });
        assert!(result.is_err());
    }

    #[test]
    fn compatibility_cpu_wrapper_fails_closed() {
        let engine = CpuEngine::new();
        let tiles = [valid_tile(), valid_tile()];
        let query = [0.0f32; codec::D_C];
        let signs = [0u64; codec::RESIDUAL_WORDS];
        let mut scores = [7.0f32; 1];
        score_tiles_cpu(ScoringInput {
            engine: &engine,
            tiles: &tiles,
            q_coarse: &query,
            q_sign: &signs,
            scores: &mut scores,
        });
        assert_eq!(scores, [0.0]);
    }
}
