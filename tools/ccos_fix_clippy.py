from pathlib import Path

p = Path("slha-c/src/elastic_ffi.rs")
s = p.read_text()

anchor = "fn tile_bytes(tile: *const SciRustSlhaTile) -> Result<[u8; 128], i32> {\n"
helpers = r'''fn drop_cache_handle(handle: *mut SlhaElasticKvCache) {
    // SAFETY: ownership of a pointer returned by `slha_elastic_cache_new` is
    // transferred back exactly once by the public free function.
    drop(unsafe { Box::from_raw(handle) });
}

fn read_f32_at(base: *const f32, index: usize) -> f32 {
    // SAFETY: the caller-facing function validates the contract that `base`
    // contains the requested readable range; unaligned C storage is accepted.
    unsafe { ptr::read_unaligned(base.add(index)) }
}

fn write_f32_at(base: *mut f32, index: usize, value: f32) {
    // SAFETY: the caller-facing function validates the contract that `base`
    // contains the requested writable range; unaligned C storage is accepted.
    unsafe { ptr::write_unaligned(base.add(index), value) };
}

fn copy_tile_out(bytes: &[u8; 128], out: *mut SciRustSlhaTile) {
    // SAFETY: the caller-facing function requires one writable ABI tile.
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), bytes.len()) };
}

fn write_stats_out(out: *mut SlhaElasticKvCacheStats, stats: SlhaElasticKvCacheStats) {
    // SAFETY: the caller-facing function requires writable storage for one
    // stats record; write_unaligned preserves the C ABI alignment contract.
    unsafe { ptr::write_unaligned(out, stats) };
}

'''
if anchor not in s:
    raise RuntimeError("missing helper insertion anchor")
s = s.replace(anchor, helpers + anchor, 1)

old = '''        // SAFETY: ownership of a pointer returned by `slha_elastic_cache_new`
        // is transferred back exactly once.
        drop(unsafe { Box::from_raw(handle) });
'''
if old not in s:
    raise RuntimeError("missing free raw-pointer block")
s = s.replace(old, "        drop_cache_handle(handle);\n", 1)

old = '''            // SAFETY: caller provides `count` writable float slots.
            unsafe { ptr::write_unaligned(scores_out.add(index), value) };
'''
if old not in s:
    raise RuntimeError("missing score output raw-pointer block")
s = s.replace(old, "            write_f32_at(scores_out, index, value);\n", 1)

old = '''            // SAFETY: caller provides `count` readable float scores.
            let score = unsafe { ptr::read_unaligned(scores.add(offset)) };
'''
if old not in s:
    raise RuntimeError("missing observed-score raw-pointer block")
s = s.replace(old, "            let score = read_f32_at(scores, offset);\n", 1)

old = '''        // SAFETY: caller provides one writable tile. Byte-wise copy accepts
        // unaligned output storage.
        unsafe {
            ptr::copy_nonoverlapping(tile.as_ptr(), out_tile.cast::<u8>(), tile.len());
        }
'''
if old not in s:
    raise RuntimeError("missing tile output raw-pointer block")
s = s.replace(old, "        copy_tile_out(&tile, out_tile);\n", 1)

old = '''        // SAFETY: caller provides writable storage for one stats value; unaligned
        // output is explicitly accepted.
        unsafe { ptr::write_unaligned(out, stats) };
'''
if old not in s:
    raise RuntimeError("missing stats output raw-pointer block")
s = s.replace(old, "        write_stats_out(out, stats);\n", 1)

p.write_text(s)
