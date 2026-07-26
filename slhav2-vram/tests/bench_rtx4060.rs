#![cfg(feature = "cuda")]

/// RTX 4060 performance benchmarks.
///
/// Measures per-token latency (μs) and effective VRAM throughput (GB/s) for
/// the low-rank TurboQuant matmul kernel. Prints a formatted table to stdout.
///
/// Requires `--features cuda`.

use std::time::Instant;

use slhav2_vram::backends::cuda_driver::CudaDriverBackend;
use slhav2_vram::traits::DeviceEngine;

/// Generate a deterministic packed TurboQuant weight buffer.
fn make_weights(dim_n: usize, dim_k: usize) -> Vec<u8> {
    let group_size = 16;
    let num_groups = dim_k / group_size;
    let packed_bytes = dim_n * (dim_k / 2);
    let scales_bytes = dim_n * num_groups * 4;
    let mut buf = vec![0u8; packed_bytes + scales_bytes];

    let mut rng: u64 = 0x1234;
    for b in buf[..packed_bytes].iter_mut() {
        rng = rng.wrapping_mul(0x9E3779B97F4A7C15);
        rng ^= rng >> 30;
        *b = rng as u8;
    }

    let scales: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(
            buf[packed_bytes..].as_mut_ptr() as *mut f32,
            dim_n * num_groups,
        )
    };
    for s in scales.iter_mut() {
        rng = rng.wrapping_mul(0xBF58476D1CE4E5B9);
        rng ^= rng >> 27;
        *s = ((rng as f64 * 2.0_f64.powi(-32)).abs() + 0.1) as f32;
    }

    buf
}

/// Benchmark configurations: (M, N, K, description).
const CONFIGS: &[(usize, usize, usize, &str)] = &[
    (1, 4096, 128, "1×4096×128  (single token, hidden=4096)"),
    (1, 8192, 128, "1×8192×128  (single token, hidden=8192)"),
    (4, 4096, 128, "4×4096×128  (4 tokens)"),
    (8, 4096, 128, "8×4096×128  (8 tokens)"),
    (1, 16384, 128, "1×16384×128 (very wide)"),
];

/// Print a formatted benchmark table to stdout.
fn print_table(results: &[(usize, usize, usize, f64, f64)]) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RTX 4060 — lowrank_turboquant_matmul Benchmarks");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  {:<36} {:>12} {:>14}",
        "Configuration", "Latency (μs)", "Throughput (GB/s)"
    );
    println!("───────────────────────────────────────────────────────────────────");
    for (m, n, k, latency_us, throughput_gbps) in results {
        let label = format!("{m}×{n}×{k}");
        println!(
            "  {:<36} {:>12.1} {:>14.2}",
            label, latency_us, throughput_gbps
        );
    }
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
}

#[test]
fn bench_rtx4060_latency_throughput() {
    let engine = CudaDriverBackend::new(0).expect("init CUDA");

    let warmup_m = 1;
    let warmup_n = 4096;
    let warmup_k = 128;

    let w_warmup = engine
        .allocate(make_weights(warmup_n, warmup_k).len())
        .expect("alloc warmup weights");

    // Warmup: 10 iterations
    {
        let inp = engine
            .allocate(warmup_m * warmup_k * 4)
            .expect("alloc warmup input");
        let out = engine
            .allocate(warmup_m * warmup_n * 4)
            .expect("alloc warmup output");

        let host_inp = vec![0.5f32; warmup_m * warmup_k];
        let inp_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(host_inp.as_ptr() as *const u8, host_inp.len() * 4)
        };
        engine.copy_to_device(inp_u8, &inp).expect("copy warmup inp");

        let w_data = make_weights(warmup_n, warmup_k);
        engine
            .copy_to_device(&w_data, &w_warmup)
            .expect("copy warmup w");

        for _ in 0..10 {
            engine
                .launch_lowrank_matmul(&inp, &w_warmup, &out, warmup_m, warmup_n, warmup_k)
                .expect("warmup matmul");
        }
        engine.synchronize().expect("warmup sync");

        engine.free(inp).expect("free warmup inp");
        engine.free(out).expect("free warmup out");
    }

    let mut results = Vec::new();

    for &(m, n, k, _desc) in CONFIGS {
        let inp_bytes = m * k * 4;
        let out_bytes = m * n * 4;
        let w_bytes = make_weights(n, k).len();

        let inp = engine.allocate(inp_bytes).expect("alloc input");
        let w = engine.allocate(w_bytes).expect("alloc weights");
        let out = engine.allocate(out_bytes).expect("alloc output");

        let host_inp = vec![0.5f32; m * k];
        let inp_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(host_inp.as_ptr() as *const u8, host_inp.len() * 4)
        };
        engine.copy_to_device(inp_u8, &inp).expect("copy input");

        let w_data = make_weights(n, k);
        engine.copy_to_device(&w_data, &w).expect("copy weights");

        engine.synchronize().expect("pre-bench sync");

        const ITERATIONS: u32 = 100;
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            engine
                .launch_lowrank_matmul(&inp, &w, &out, m, n, k)
                .expect("bench matmul");
        }
        engine.synchronize().expect("bench sync");
        let elapsed = start.elapsed();

        let avg_latency_us = elapsed.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;

        let read_bytes = (m * k * 4) as f64 + (n * (k / 2) + n * (k / 16) * 4) as f64;
        let write_bytes = (m * n * 4) as f64;
        let total_bytes = read_bytes + write_bytes;

        let throughput_gbps = total_bytes / (avg_latency_us * 1e-6) / 1e9;

        results.push((m, n, k, avg_latency_us, throughput_gbps));

        engine.free(inp).expect("free input");
        engine.free(w).expect("free weights");
        engine.free(out).expect("free output");
    }

    engine.free(w_warmup).expect("free warmup weights");

    print_table(&results);

    let (_m, _n, _k, lat, _tput) = results[0];
    assert!(
        lat < 500.0,
        "1×4096×128 latency {:.1}μs exceeds 500μs threshold",
        lat
    );
}
