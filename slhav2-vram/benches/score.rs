use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use slhav2_vram::codec;
use slhav2_vram::mem::tile::SerializedTile;
use slhav2_vram::pipeline::{score_tiles_cpu, ScoringInput};

fn build_tiles(n: usize) -> Vec<SerializedTile> {
    let mut tiles = Vec::with_capacity(n);
    for i in 0..n {
        let mut t = SerializedTile::zeroed();
        t.set_scale(1.0);
        t.set_dynamic_lambda(0.05);
        t.set_group_scales(&[128u8; 8]);
        t.set_flags(0);
        for d in 0..codec::LATENT_KV_WORDS {
            let lo = ((i + d) as u8) & 0x0F;
            let hi = (((i + d) >> 4) as u8) & 0x0F;
            t.as_bytes_mut()[d] = (hi << 4) | lo;
        }
        tiles.push(t);
    }
    tiles
}

fn bench_score_tiles_cpu(c: &mut Criterion) {
    let tiles = build_tiles(256);
    let q_coarse: Vec<f32> = (0..codec::D_C).map(|i| (i as f32) * 0.01).collect();
    let q_sign: Vec<u64> = vec![0xABCDEF0123456789u64; codec::RESIDUAL_WORDS];
    let mut scores = vec![0.0f32; tiles.len()];

    let engine = slhav2_vram::backends::cpu::CpuEngine::new();

    c.bench_function("score_256_tiles_cpu", |b| {
        b.iter(|| {
            score_tiles_cpu(ScoringInput {
                engine: &engine,
                tiles: black_box(&tiles),
                q_coarse: black_box(&q_coarse),
                q_sign: black_box(&q_sign),
                scores: black_box(&mut scores),
            });
        })
    });
}

criterion_group!(benches, bench_score_tiles_cpu);
criterion_main!(benches);
