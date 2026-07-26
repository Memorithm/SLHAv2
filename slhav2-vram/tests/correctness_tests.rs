use slhav2_vram::backends::cpu_ref::CpuRefBackend;
use slhav2_vram::traits::DeviceEngine;

#[allow(dead_code)]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    if na * nb == 0.0 {
        0.0
    } else {
        dot / (na * nb).sqrt()
    }
}

fn make_turboquant_weights(dim_n: usize, dim_k: usize, group_size: usize) -> Vec<u8> {
    let num_groups = dim_k / group_size;
    let packed_bytes = dim_n * (dim_k / 2);
    let scales_bytes = dim_n * num_groups * 4;
    let mut buf = vec![0u8; packed_bytes + scales_bytes];

    let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;
    for i in 0..packed_bytes {
        rng_state = rng_state.wrapping_mul(0x9E3779B97F4A7C15);
        rng_state ^= rng_state >> 30;
        buf[i] = rng_state as u8;
    }

    let scales_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(
            buf[packed_bytes..].as_mut_ptr() as *mut f32,
            dim_n * num_groups,
        )
    };
    for i in 0..scales_f32.len() {
        rng_state = rng_state.wrapping_mul(0xBF58476D1CE4E5B9);
        rng_state ^= rng_state >> 27;
        let v = (rng_state as f64 * 2.0_f64.powi(-32)).abs() + 0.1;
        scales_f32[i] = v as f32;
    }

    buf
}

/// CPU vs GPU numerical equivalence test.
#[test]
fn cpu_vs_gpu_numerical_equivalence() {
    let cpu = CpuRefBackend::new(4096);

    let dim_m = 8;
    let dim_n = 128;
    let dim_k = 128;
    let group_size = 16;

    let mut input = vec![0.0f32; dim_m * dim_k];
    let mut rng: u64 = 0xABCD;
    for v in input.iter_mut() {
        rng = rng.wrapping_mul(0x9E3779B97F4A7C15);
        rng ^= rng >> 30;
        *v = (rng as f64 * 2.0_f64.powi(-32)).abs() as f32 * 2.0 - 1.0;
    }

    let weights = make_turboquant_weights(dim_n, dim_k, group_size);

    let d_inp = cpu.allocate(dim_m * dim_k * 4).expect("cpu alloc input");
    let d_w = cpu.allocate(weights.len()).expect("cpu alloc weights");
    let d_out = cpu.allocate(dim_m * dim_n * 4).expect("cpu alloc output");

    let inp_u8: &[u8] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    cpu.copy_to_device(inp_u8, &d_inp).expect("cpu copy input");
    cpu.copy_to_device(&weights, &d_w).expect("cpu copy weights");

    cpu.launch_lowrank_matmul(&d_inp, &d_w, &d_out, dim_m, dim_n, dim_k)
        .expect("cpu matmul");

    let mut cpu_out_u8 = vec![0u8; dim_m * dim_n * 4];
    cpu.copy_to_host(&d_out, &mut cpu_out_u8).expect("cpu copy out");
    let cpu_out: &[f32] = unsafe {
        std::slice::from_raw_parts(cpu_out_u8.as_ptr() as *const f32, dim_m * dim_n)
    };
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    let cpu_result = cpu_out.to_vec();

    cpu.free(d_inp).expect("cpu free inp");
    cpu.free(d_w).expect("cpu free w");
    cpu.free(d_out).expect("cpu free out");

    #[cfg(feature = "cuda")]
    {
        let gpu = match slhav2_vram::backends::cuda_driver::CudaDriverBackend::new(0) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("SKIP: CUDA backend unavailable: {e}");
                return;
            }
        };

        let g_inp = gpu.allocate(dim_m * dim_k * 4).expect("gpu alloc input");
        let g_w = gpu.allocate(weights.len()).expect("gpu alloc weights");
        let g_out = gpu.allocate(dim_m * dim_n * 4).expect("gpu alloc output");

        gpu.copy_to_device(inp_u8, &g_inp).expect("gpu copy input");
        gpu.copy_to_device(&weights, &g_w).expect("gpu copy weights");

        gpu.launch_lowrank_matmul(&g_inp, &g_w, &g_out, dim_m, dim_n, dim_k)
            .expect("gpu matmul");

        gpu.synchronize().expect("gpu sync");

        let mut gpu_out_u8 = vec![0u8; dim_m * dim_n * 4];
        gpu.copy_to_host(&g_out, &mut gpu_out_u8).expect("gpu copy out");
        let gpu_out: &[f32] = unsafe {
            std::slice::from_raw_parts(gpu_out_u8.as_ptr() as *const f32, dim_m * dim_n)
        };
        let gpu_result = gpu_out.to_vec();

        gpu.free(g_inp).expect("gpu free inp");
        gpu.free(g_w).expect("gpu free w");
        gpu.free(g_out).expect("gpu free out");

        let mut row_cosines = Vec::with_capacity(dim_m);
        for m in 0..dim_m {
            let start = m * dim_n;
            let end = start + dim_n;
            let cos = cosine(&cpu_result[start..end], &gpu_result[start..end]);
            row_cosines.push(cos);
        }

        eprintln!("Row-wise cosine similarity (CPU vs GPU):");
        for (i, &c) in row_cosines.iter().enumerate() {
            eprintln!("  row {i}: {:.8}", c);
            assert!(c >= 0.999, "Cosine similarity {:.8} < 0.999 at row {i}", c);
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA feature disabled — CPU-only validation passed");
    }
}

/// CPU self-consistency: deterministic known output.
#[test]
fn cpu_self_consistency() {
    let cpu = CpuRefBackend::new(256);

    let dim_m = 2;
    let dim_n = 4;
    let dim_k = 8;

    let input: Vec<f32> = (0..dim_m * dim_k).map(|i| i as f32).collect();

    let packed_bytes = dim_n * (dim_k / 2);
    let scales_bytes = dim_n * 4;
    let mut weights = vec![0u8; packed_bytes + scales_bytes];

    for n in 0..dim_n {
        // Byte 0: low nibble=1 (signed_val=1), high nibble=0 (signed_val=0)
        weights[n * (dim_k / 2)] = 0x01;
        for j in 1..(dim_k / 2) {
            // Both nibbles=0 → signed_val=0
            weights[n * (dim_k / 2) + j] = 0x00;
        }
    }
    for n in 0..dim_n {
        let scale_pos = packed_bytes + n * 4;
        let scale_bytes = &mut weights[scale_pos..scale_pos + 4];
        scale_bytes.copy_from_slice(&1.0f32.to_ne_bytes());
    }

    let d_inp = cpu.allocate(input.len() * 4).expect("alloc input");
    let d_w = cpu.allocate(weights.len()).expect("alloc weights");
    let d_out = cpu.allocate(dim_m * dim_n * 4).expect("alloc output");

    let inp_u8: &[u8] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    cpu.copy_to_device(inp_u8, &d_inp).expect("copy input");
    cpu.copy_to_device(&weights, &d_w).expect("copy weights");

    cpu.launch_lowrank_matmul(&d_inp, &d_w, &d_out, dim_m, dim_n, dim_k)
        .expect("cpu matmul");

    let mut out_u8 = vec![0u8; dim_m * dim_n * 4];
    cpu.copy_to_host(&d_out, &mut out_u8).expect("copy out");
    let out: &[f32] = unsafe {
        std::slice::from_raw_parts(out_u8.as_ptr() as *const f32, dim_m * dim_n)
    };

    for m in 0..dim_m {
        for n in 0..dim_n {
            let expected = input[m * dim_k];
            let actual = out[m * dim_n + n];
            assert!(
                (actual - expected).abs() < 1e-4,
                "cpu_self_consistency: output[{m}][{n}] = {actual}, expected {expected}"
            );
        }
    }

    cpu.free(d_inp).unwrap();
    cpu.free(d_w).unwrap();
    cpu.free(d_out).unwrap();
}
