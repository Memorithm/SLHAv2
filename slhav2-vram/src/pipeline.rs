use crate::codec;
use crate::mem::tile::SerializedTile;
use crate::traits::{DeviceAllocation, DeviceEngine};

pub struct TileBatch<'a, A: DeviceAllocation> {
    pub tiles_dev: &'a A,
    pub scores_dev: &'a A,
    pub num_tiles: usize,
}

pub struct QueryData {
    pub q_coarse: Vec<f32>,
    pub q_sign: Vec<u64>,
}

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
    let total = tiles.len() * codec::TILE_BYTES;
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
    let total = num_scores * 4;
    let mut buf = vec![0u8; total];
    engine.copy_to_host(src_alloc, offset, &mut buf)?;
    let scores: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(scores)
}
