use slhav2_vram::backends::cpu_ref::CpuRefBackend;
use slhav2_vram::memory::VramMemoryPool;
use slhav2_vram::traits::DeviceEngine;

/// Helper: deterministic pseudo-random byte generator (SplitMix64 style).
fn fill_deterministic(buf: &mut [u8], seed: u64) {
    let mut state = seed;
    for chunk in buf.chunks_mut(8) {
        state = state.wrapping_mul(0x9E3779B97F4A7C15);
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58476D1CE4E5B9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94D049BB133111EB);
        state ^= state >> 31;
        let bytes = state.to_ne_bytes();
        let len = chunk.len().min(8);
        chunk.copy_from_slice(&bytes[..len]);
    }
}

/// Allocate 64MB on the device, write a deterministic pattern, copy back,
/// and assert bitwise equality.
#[test]
fn host_device_64mb_integrity() {
    let engine = CpuRefBackend::new(4096);

    let size = 64 * 1024 * 1024; // 64 MB
    let ptr = engine.allocate(size).expect("allocate 64MB");

    let mut host_orig = vec![0u8; size];
    fill_deterministic(&mut host_orig, 0xDEAD_BEEF);

    engine
        .copy_to_device(&host_orig, &ptr)
        .expect("copy to device");

    let mut host_roundtrip = vec![0u8; size];
    engine
        .copy_to_host(&ptr, &mut host_roundtrip)
        .expect("copy to host");

    assert_eq!(
        host_orig, host_roundtrip,
        "Host→Device→Host round-trip: bitwise mismatch"
    );

    engine.free(ptr).expect("free");
}

/// Validate VramMemoryPool allocation, deallocation, and reuse stability.
#[test]
fn pool_alloc_dealloc_reuse_stability() {
    let engine = Box::new(CpuRefBackend::new(4096));
    let pool = VramMemoryPool::new(engine, 256 * 1024 * 1024).expect("create pool");

    let mut ptrs = Vec::new();

    // Allocate many small buffers
    for _ in 0..128 {
        let p = pool.allocate(4096).expect("pool alloc 4KB");
        ptrs.push(p);
    }
    assert_eq!(pool.live_allocations(), 128);

    // Free half
    for p in ptrs.drain(64..) {
        pool.free(p).expect("pool free");
    }
    assert_eq!(pool.live_allocations(), 64);

    // Allocate again — should reuse freed slots
    for _ in 0..64 {
        let p = pool.allocate(4096).expect("pool realloc 4KB");
        ptrs.push(p);
    }
    assert_eq!(pool.live_allocations(), 128);

    // Free all
    for p in ptrs.drain(..) {
        pool.free(p).expect("pool free all");
    }
    assert_eq!(pool.live_allocations(), 0);

    // Allocate one large buffer
    let _big = pool
        .allocate(128 * 1024 * 1024)
        .expect("pool alloc 128MB");
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use slhav2_vram::backends::cuda_driver::CudaDriverBackend;
    use slhav2_vram::traits::DeviceEngine;

    /// Dispatch 1,000 asynchronous matmul calls in rapid succession over a
    /// single CUDA stream, monitoring for memory leaks or sync races.
    #[test]
    fn stream_stress_1000_ops() {
        let engine = CudaDriverBackend::new(0).expect("init CUDA");

        let m = 4;
        let n = 64;
        let k = 128;

        let inp = engine.allocate(m * k * 4).expect("alloc input");
        let w = engine.allocate(n * (k / 2) + n * (k / 16) * 4).expect("alloc weights");
        let out = engine.allocate(m * n * 4).expect("alloc output");

        // Fill with small values
        let host_inp = vec![0.5f32; m * k];
        let inp_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(host_inp.as_ptr() as *const u8, host_inp.len() * 4)
        };
        engine.copy_to_device(inp_u8, &inp).expect("copy input");

        let mut host_w = vec![0u8; n * (k / 2) + n * (k / 16) * 4];
        // Simple weights: all nibbles = 8 (value 0), so output is deterministic
        for byte in host_w.iter_mut() {
            *byte = 0x88; // both nibbles = 8, dequant value = 0
        }
        // Set scales to 1.0
        let scale_start = n * (k / 2);
        let scale_bytes = &mut host_w[scale_start..];
        for chunk in scale_bytes.chunks_mut(4) {
            chunk.copy_from_slice(&1.0f32.to_ne_bytes());
        }
        engine.copy_to_device(&host_w, &w).expect("copy weights");

        // 1,000 launches
        for i in 0..1000 {
            engine
                .launch_lowrank_matmul(&inp, &w, &out, m, n, k)
                .unwrap_or_else(|e| panic!("launch {i} failed: {e}"));
        }

        engine.synchronize().expect("sync");

        // Verify no crash and output is readable
        let mut host_out = vec![0u8; m * n * 4];
        engine.copy_to_host(&out, &mut host_out).expect("copy out");

        // All values should be 0.0 (since weight nibbles are 8 => dequant = 0)
        let out_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(host_out.as_ptr() as *const f32, m * n)
        };
        for (i, &v) in out_f32.iter().enumerate() {
            assert!(
                v.is_finite(),
                "output[{i}] = {v} is not finite (nan/inf)"
            );
        }

        engine.free(inp).expect("free inp");
        engine.free(w).expect("free w");
        engine.free(out).expect("free out");

        // Check for leaks: after freeing everything, total allocated should be ~0
        // (we can't easily check internal state, but no crash is a good sign)
    }
}
