from pathlib import Path

p = Path('slha-c/src/elastic_ffi.rs')
s = p.read_text()
s = s.replace(
    'use slhav2_vram::elastic_cache::{ElasticKvCache, PhysicalTier};\n',
    'use slhav2_vram::{\n    codec,\n    elastic_cache::{ElasticKvCache, PhysicalTier},\n};\n',
    1,
)

marker = '/// Create a registry-backed elastic fixed-slot KV cache.\n'
insert = '''/// Resident bytes charged by one full HOT tile.\n#[no_mangle]\npub extern "C" fn slha_elastic_hot_resident_bytes() -> usize {\n    codec::HOT_BYTES\n}\n\n/// Resident bytes charged by one WARM tile.\n#[no_mangle]\npub extern "C" fn slha_elastic_warm_resident_bytes() -> usize {\n    codec::WARM_BYTES\n}\n\nfn strided_slot(start_slot: usize, stride: usize, offset: usize) -> Result<usize, FfiError> {\n    let delta = stride.checked_mul(offset).ok_or_else(|| {\n        ffi_error(\n            SLHA_ERR_DIMENSION,\n            "elastic strided slot multiplication overflows usize",\n        )\n    })?;\n    start_slot.checked_add(delta).ok_or_else(|| {\n        ffi_error(\n            SLHA_ERR_DIMENSION,\n            "elastic strided slot addition overflows usize",\n        )\n    })\n}\n\n'''
if marker not in s:
    raise RuntimeError('missing elastic cache constructor marker')
s = s.replace(marker, insert + marker, 1)

marker = '/// Update slot importance from a contiguous range of attention scores.\n'
insert = '''/// Score a fixed-slot arithmetic progression without partially modifying\n/// output on failure. This lets an engine interleave layers by physical KV\n/// position without materializing a dense `layers * context` backing array.\n///\n/// # Safety\n/// `q_coarse` points to `D_C` readable f32 values, `q_sign` points to\n/// `RESIDUAL_WORDS` readable u64 values and `scores_out` points to `count`\n/// writable f32 values. Unaligned storage is accepted.\n#[no_mangle]\npub unsafe extern "C" fn slha_elastic_cache_score_strided(\n    handle: *mut SlhaElasticKvCache,\n    start_slot: usize,\n    stride: usize,\n    count: usize,\n    q_coarse: *const f32,\n    q_sign: *const u64,\n    scores_out: *mut f32,\n) -> i32 {\n    ffi_status(|| {\n        let cache = cache_arc(handle)?;\n        if count > MAX_TILES {\n            return Err(ffi_error(\n                SLHA_ERR_DIMENSION,\n                format!("score count {count} exceeds safety bound {MAX_TILES}"),\n            ));\n        }\n        if count == 0 {\n            return Ok(());\n        }\n        if count > 1 && stride == 0 {\n            return Err(ffi_error(\n                SLHA_ERR_DIMENSION,\n                "elastic strided score requires non-zero stride when count > 1",\n            ));\n        }\n        if scores_out.is_null() {\n            return Err(ffi_error(SLHA_ERR_NULL, "score output pointer is NULL"));\n        }\n        // SAFETY: guaranteed by this function's caller contract.\n        let (coarse, sign) = unsafe { read_query(q_coarse, q_sign) }?;\n        let guard = lock_cache(&cache);\n        let mut values = Vec::new();\n        values\n            .try_reserve_exact(count)\n            .map_err(|_| ffi_error(SLHA_ERR_PANIC, "score buffer allocation failed"))?;\n        for offset in 0..count {\n            let slot = strided_slot(start_slot, stride, offset)?;\n            values.push(guard.score(slot, &coarse, &sign).ok_or_else(|| {\n                ffi_error(\n                    SLHA_ERR_NOT_RESIDENT,\n                    format!("elastic cache slot {slot} is absent or COLD"),\n                )\n            })?);\n        }\n        drop(guard);\n        // SAFETY: output is validated above and caller guarantees `count` slots.\n        unsafe { write_scores(scores_out, &values) };\n        Ok(())\n    })\n}\n\n'''
if marker not in s:
    raise RuntimeError('missing observe marker')
s = s.replace(marker, insert + marker, 1)

marker = '/// Transactionally demote resident slots toward a target residency.\n'
insert = '''/// Update slot importance over a fixed-slot arithmetic progression.\n///\n/// # Safety\n/// `scores` points to `count` readable f32 values. Unaligned storage is\n/// accepted. `temperature` must be finite and strictly positive.\n#[no_mangle]\npub unsafe extern "C" fn slha_elastic_cache_observe_scores_strided(\n    handle: *mut SlhaElasticKvCache,\n    start_slot: usize,\n    stride: usize,\n    scores: *const f32,\n    count: usize,\n    temperature: f32,\n) -> i32 {\n    ffi_status(|| {\n        let cache = cache_arc(handle)?;\n        if !temperature.is_finite() || temperature <= 0.0 {\n            return Err(ffi_error(\n                SLHA_ERR_DIMENSION,\n                "elastic score temperature must be finite and positive",\n            ));\n        }\n        if count > MAX_TILES {\n            return Err(ffi_error(\n                SLHA_ERR_DIMENSION,\n                format!("score count {count} exceeds safety bound {MAX_TILES}"),\n            ));\n        }\n        if count == 0 {\n            return Ok(());\n        }\n        if count > 1 && stride == 0 {\n            return Err(ffi_error(\n                SLHA_ERR_DIMENSION,\n                "elastic strided observation requires non-zero stride when count > 1",\n            ));\n        }\n        // SAFETY: guaranteed by this function's caller contract.\n        let values = unsafe { read_scores(scores, count) }?;\n        let mut observations = Vec::new();\n        observations\n            .try_reserve_exact(count)\n            .map_err(|_| ffi_error(SLHA_ERR_PANIC, "importance buffer allocation failed"))?;\n        for (offset, score) in values.into_iter().enumerate() {\n            observations.push((strided_slot(start_slot, stride, offset)?, score));\n        }\n        lock_cache(&cache).observe_scores(&observations, temperature);\n        Ok(())\n    })\n}\n\n'''
if marker not in s:
    raise RuntimeError('missing demote marker')
s = s.replace(marker, insert + marker, 1)

marker = '    #[test]\n    fn forged_and_released_handles_are_rejected_without_dereference() {\n'
insert = '''    #[test]\n    fn resident_byte_constants_match_physical_codec() {\n        assert_eq!(slha_elastic_hot_resident_bytes(), codec::HOT_BYTES);\n        assert_eq!(slha_elastic_warm_resident_bytes(), codec::WARM_BYTES);\n        assert_eq!(slha_elastic_hot_resident_bytes(), 128);\n        assert_eq!(slha_elastic_warm_resident_bytes(), 96);\n    }\n\n    #[test]\n    fn strided_scoring_addresses_one_interleaved_layer_transactionally() {\n        let handle = slha_elastic_cache_new(4096);\n        assert!(!handle.is_null());\n        let mut first = tile();\n        first.latent_kv.fill(0x88);\n        let mut gap = tile();\n        gap.latent_kv.fill(0x99);\n        let mut second = tile();\n        second.latent_kv.fill(0x88);\n        assert_eq!(write(handle, 1, &first), SLHA_OK);\n        assert_eq!(write(handle, 2, &gap), SLHA_OK);\n        assert_eq!(write(handle, 4, &second), SLHA_OK);\n\n        let q_coarse = [1.0f32; D_C];\n        let q_sign = [0u64; RESIDUAL_WORDS];\n        let mut output = [-7.0f32; 2];\n        // SAFETY: all buffers are local and sized according to the ABI contract.\n        let rc = unsafe {\n            slha_elastic_cache_score_strided(\n                handle,\n                1,\n                3,\n                2,\n                q_coarse.as_ptr(),\n                q_sign.as_ptr(),\n                output.as_mut_ptr(),\n            )\n        };\n        assert_eq!(rc, SLHA_OK);\n        assert_eq!(output[0], output[1]);\n\n        let observations = [4.0f32, 1.0f32];\n        // SAFETY: the score buffer is local and contains exactly two elements.\n        assert_eq!(\n            unsafe {\n                slha_elastic_cache_observe_scores_strided(\n                    handle,\n                    1,\n                    3,\n                    observations.as_ptr(),\n                    observations.len(),\n                    1.0,\n                )\n            },\n            SLHA_OK\n        );\n\n        output.fill(123.0);\n        assert_eq!(slha_elastic_cache_clear_slot(handle, 4), SLHA_OK);\n        // SAFETY: all buffers remain valid; the missing second slot must leave\n        // the complete output unchanged.\n        assert_eq!(\n            unsafe {\n                slha_elastic_cache_score_strided(\n                    handle,\n                    1,\n                    3,\n                    2,\n                    q_coarse.as_ptr(),\n                    q_sign.as_ptr(),\n                    output.as_mut_ptr(),\n                )\n            },\n            SLHA_ERR_NOT_RESIDENT\n        );\n        assert_eq!(output, [123.0, 123.0]);\n        assert_eq!(slha_elastic_cache_free(handle), SLHA_OK);\n    }\n\n'''
if marker not in s:
    raise RuntimeError('missing FFI test insertion marker')
s = s.replace(marker, insert + marker, 1)
p.write_text(s)

p = Path('slha-c/include/slha.h')
s = p.read_text()
s = s.replace(
    'SlhaElasticKvCache* slha_elastic_cache_new(size_t hard_budget_bytes);\n',
    'size_t slha_elastic_hot_resident_bytes(void);\nsize_t slha_elastic_warm_resident_bytes(void);\nSlhaElasticKvCache* slha_elastic_cache_new(size_t hard_budget_bytes);\n',
    1,
)
needle = '''int32_t slha_elastic_cache_score_range(\n    SlhaElasticKvCache* cache,\n    size_t start_slot,\n    size_t count,\n    const float* q_coarse,\n    const uint64_t* q_sign,\n    float* scores_out\n);\n'''
addition = needle + '''int32_t slha_elastic_cache_score_strided(\n    SlhaElasticKvCache* cache,\n    size_t start_slot,\n    size_t stride,\n    size_t count,\n    const float* q_coarse,\n    const uint64_t* q_sign,\n    float* scores_out\n);\n'''
if needle not in s:
    raise RuntimeError('missing score range declaration')
s = s.replace(needle, addition, 1)
needle = '''int32_t slha_elastic_cache_observe_scores(\n    SlhaElasticKvCache* cache,\n    size_t start_slot,\n    const float* scores,\n    size_t count,\n    float temperature\n);\n'''
addition = needle + '''int32_t slha_elastic_cache_observe_scores_strided(\n    SlhaElasticKvCache* cache,\n    size_t start_slot,\n    size_t stride,\n    const float* scores,\n    size_t count,\n    float temperature\n);\n'''
if needle not in s:
    raise RuntimeError('missing observe declaration')
s = s.replace(needle, addition, 1)
p.write_text(s)
