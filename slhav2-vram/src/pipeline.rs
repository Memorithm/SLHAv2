use crate::codec;
use crate::mem::tile::SerializedTile;
use crate::traits::{DeviceAllocation, DeviceEngine};

pub struct ScoringInput<'a, E: DeviceEngine> {
    pub engine: &'a E,
    pub tiles: &'a [SerializedTile],
    pub q_coarse: &'a [f32],
    pub q_sign: &'a [u64],
    pub scores: &'a mut [f32],
}

pub fn score_tiles_cpu(input: ScoringInput<'_, impl DeviceEngine<Alloc = impl DeviceAllocation>>) {
    let q_coarse = input.q_coarse;
    let q_sign = input.q_sign;
    let scores = input.scores;

    for (i, tile) in input.tiles.iter().enumerate() {
        scores[i] = tile.score(q_coarse, q_sign);
    }
}

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
    let mut buf = vec![0u8; total];
    for (i, tile) in tiles.iter().enumerate() {
        let off = i * codec::TILE_BYTES;
        buf[off..off + codec::TILE_BYTES].copy_from_slice(&tile.0);
    }
    engine.copy_to_device(&buf, dst_alloc, offset)
}

pub fn copy_scores_from_gpu<E: DeviceEngine>(
    engine: &E,
    src_alloc: &E::Alloc,
    offset: usize,
    num_scores: usize,
) -> Result<Vec<f32>, E::Error> {
    let total = num_scores
        .checked_mul(4)
        .expect("copy_scores_from_gpu: byte count overflow");
    let mut buf = vec![0u8; total];
    engine.copy_to_host(src_alloc, offset, &mut buf)?;
    let scores = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(scores)
}

/// A persistent GPU scoring pipeline for the CUDA backend.
///
/// The convenience helpers ([`copy_tiles_to_gpu`]/[`copy_scores_from_gpu`])
/// re-upload the entire tile set and re-copy all scores on every call — for a
/// growing KV cache scored every decode step that is O(cache) PCIe traffic
/// per token, which negates the point of a device-resident arena.
///
/// This pipeline owns one persistent device arena plus a per-step staging
/// buffer. It tracks which tiles changed since the last upload (a dirty
/// bitmap) and issues only the dirty subset to the device on a stream,
/// overlapping the H2D copy, the kernel launch and the D2H copy. The final
/// sync is a single `cuStreamSynchronize` instead of a context-wide
/// `cuCtxSynchronize` per batch.
///
/// Construction is hardware-bound (requires a live `cuda::CudaEngine`, a
/// loaded module kernel and a device arena of `capacity_bytes`); see the CUDA
/// integration tests for a full usage example.
#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
pub struct GpuScoringPipeline {
    engine: crate::backends::cuda::CudaEngine,
    kernel: crate::backends::cuda::CudaFunction,
    stream: crate::backends::cuda::CudaStream,
    /// Persistent device arena of serialized tiles.
    tiles_dev: crate::backends::cuda::CudaAllocation,
    /// Persistent device score buffer (one f32 per tile slot).
    scores_dev: crate::backends::cuda::CudaAllocation,
    q_coarse_dev: crate::backends::cuda::CudaAllocation,
    q_sign_dev: crate::backends::cuda::CudaAllocation,
    /// Host-side mirror of the arena, for dirty comparison.
    host_tiles: Vec<u8>,
    /// Per-tile dirty bits (1 = needs upload).
    dirty: Vec<bool>,
    /// Number of tiles currently resident.
    resident: usize,
    /// Host staging buffer for the dirty subset.
    staging: Vec<u8>,
    /// Host staging buffer for scores.
    score_buf: Vec<u8>,
    /// Persistent q_coarse staging (D_C f32s).
    q_coarse_buf: Vec<u8>,
    /// Persistent q_sign staging (RESIDUAL_WORDS u64s).
    q_sign_buf: Vec<u8>,
}

#[cfg(all(feature = "cuda", slhav2_cuda_ptx))]
impl GpuScoringPipeline {
    /// Create a pipeline with a persistent device arena of `capacity_bytes`.
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
        let tiles_dev = engine.allocate(capacity_bytes)?;
        let scores_dev = engine.allocate(max_tiles * 4)?;
        let q_coarse_dev = engine.allocate(codec::D_C * 4)?;
        let q_sign_dev = engine.allocate(codec::RESIDUAL_WORDS * 8)?;
        let stream = crate::backends::cuda::CudaStream::new(engine)?;
        Ok(Self {
            engine: engine.clone(),
            kernel: kernel.clone(),
            stream,
            tiles_dev,
            scores_dev,
            q_coarse_dev,
            q_sign_dev,
            host_tiles: vec![0u8; capacity_bytes],
            dirty: vec![false; max_tiles],
            resident: 0,
            staging: vec![0u8; codec::TILE_BYTES],
            score_buf: Vec::with_capacity(max_tiles * 4),
            q_coarse_buf: vec![0u8; codec::D_C * 4],
            q_sign_buf: vec![0u8; codec::RESIDUAL_WORDS * 8],
        })
    }

    /// Upload (or refresh) tiles into the persistent arena, uploading only the
    /// dirty subset. Returns the slot of the first uploaded tile.
    pub fn upload(
        &mut self,
        tiles: &[SerializedTile],
    ) -> Result<usize, crate::backends::cuda::CudaError> {
        if tiles.is_empty() {
            return Ok(0);
        }
        let start = self.resident;
        let needed = start + tiles.len();
        if needed > self.dirty.len() {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: arena holds {} tiles, need {needed}",
                self.dirty.len()
            )));
        }

        for (i, t) in tiles.iter().enumerate() {
            let slot = start + i;
            let off = slot * codec::TILE_BYTES;
            // Compare against the host mirror; upload only if changed.
            if self.host_tiles[off..off + codec::TILE_BYTES] != t.0 {
                self.staging.copy_from_slice(&t.0);
                self.engine
                    .copy_to_device_at(&self.staging, &mut self.tiles_dev, off)?;
                self.host_tiles[off..off + codec::TILE_BYTES].copy_from_slice(&t.0);
            }
            self.dirty[slot] = false;
        }
        self.resident = needed;
        Ok(start)
    }

    /// Score all resident tiles on the pipeline's stream, then sync. `scores`
    /// must have at least `self.resident` entries.
    pub fn score_into(
        &mut self,
        q_coarse: &[f32],
        q_sign: &[u64],
        scores: &mut [f32],
    ) -> Result<(), crate::backends::cuda::CudaError> {
        if self.resident == 0 {
            return Ok(());
        }
        if scores.len() < self.resident {
            return Err(crate::backends::cuda::CudaError(format!(
                "GpuScoringPipeline: scores buffer has {} entries, need {}",
                scores.len(),
                self.resident
            )));
        }

        // Refresh q on the device (query changes every step).
        self.q_coarse_buf.clear();
        for &v in q_coarse {
            self.q_coarse_buf.extend_from_slice(&v.to_le_bytes());
        }
        self.q_sign_buf.clear();
        for &v in q_sign {
            self.q_sign_buf.extend_from_slice(&v.to_le_bytes());
        }
        self.engine
            .copy_to_device_at(&self.q_coarse_buf, &mut self.q_coarse_dev, 0)?;
        self.engine
            .copy_to_device_at(&self.q_sign_buf, &mut self.q_sign_dev, 0)?;

        // Launch on the stream.
        self.engine.score_tiles_on_stream(
            &self.q_coarse_dev,
            &self.q_sign_dev,
            &self.tiles_dev,
            &self.scores_dev,
            self.resident as i32,
            &self.kernel,
            &self.stream,
        )?;

        // Async copy the scores back, then sync once.
        let nbytes = self.resident * 4;
        self.score_buf.resize(nbytes, 0);
        self.engine
            .copy_to_host_async(&self.scores_dev, 0, &mut self.score_buf, &self.stream)?;
        self.stream.synchronize()?;

        for (i, c) in self.score_buf.chunks_exact(4).enumerate() {
            scores[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(())
    }

    /// Number of tiles currently resident in the persistent arena.
    pub fn resident(&self) -> usize {
        self.resident
    }
}
