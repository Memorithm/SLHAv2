#!/usr/bin/env python3
"""Apply the remaining large-file P0 fixes against the exact audited HEAD.

Every transformation is asserted to match exactly once. Any source drift aborts
before the workflow commits, so this script can never silently apply a partial
large-file patch to the llama.cpp bridge or CUDA backend.
"""

from __future__ import annotations

import re
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched: {label}")


def regex_once(path: Path, pattern: str, replacement: str, label: str) -> None:
    text = path.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one regex match, found {count}")
    path.write_text(updated)
    print(f"patched: {label}")


# ---------------------------------------------------------------------------
# llama.cpp bridge: remove the SLHA-specific 16k cap, derive storage from the
# real runtime capacity, make allocation arithmetic fail closed, and never
# return a pointer into mutable store storage after releasing the mutex.
# ---------------------------------------------------------------------------
cpp = Path("integration/llama.cpp/shim/slha_llama.cpp")

replace_once(
    cpp,
    "#include <mutex>\n#include <random>\n",
    "#include <mutex>\n#include <new>\n#include <random>\n#include <stdexcept>\n",
    "llama allocation exception headers",
)

replace_once(
    cpp,
    '''            // Capacity per layer: positions are bounded by the KV cache the\n            // runtime actually allocates. 16384 remains the documented\n            // maximum supported by this tile-store layout; positions beyond\n            // it fail closed (write returns false, no silent truncation).\n            const size_t capacity = 16384;\n            if (!g_slha_tile_store.init(n_layers, capacity, slha_tile_size())) {\n                std::cerr << "[SLHA] failed to initialize tile store\\n";\n                return -1;\n            }\n''',
    '''            // Runtime context capacity is not known at global init. Keep only\n            // model-derived layer count and tile geometry here; the K-cache\n            // allocation hook sizes the store from llama.cpp's actual\n            // `n_tokens`. There is deliberately no SLHA-specific 16k/32k/128k\n            // context cap.\n            if (!g_slha_tile_store.init(n_layers, 0, slha_tile_size())) {\n                std::cerr << "[SLHA] failed to initialize empty tile store metadata\\n";\n                return -1;\n            }\n''',
    "llama remove hard-coded 16384 capacity",
)

replace_once(
    cpp,
    '''bool slha_tile_store::init(size_t n_layers_, size_t capacity_, size_t tile_bytes_) {\n    std::lock_guard<std::mutex> lock(mutex);\n    n_layers = n_layers_;\n    capacity = capacity_;\n    tile_bytes = tile_bytes_;\n    if (tile_bytes == 0) {\n        return false;\n    }\n    // 128-aligned tile region: allocate TILE_ALIGN-1 extra bytes and point\n    // the tile base at the first 128-aligned offset. Every element of the\n    // store then satisfies the `SciRustSlhaTile` alignment regardless of\n    // vector growth (the buffer never reallocates after init).\n    const size_t total = n_layers * capacity * tile_bytes;\n    tiles.assign(total + TILE_ALIGN - 1, 0);\n    const uintptr_t raw = reinterpret_cast<uintptr_t>(tiles.data());\n    const size_t pad = (TILE_ALIGN - (raw % TILE_ALIGN)) % TILE_ALIGN;\n    tile_base_offset = pad;\n    valid.assign(n_layers * capacity, 0);\n    return true;\n}\n''',
    '''namespace {\n\nbool slha_checked_store_geometry(\n    size_t n_layers,\n    size_t capacity,\n    size_t tile_bytes,\n    size_t * slots_out,\n    size_t * allocation_out\n) {\n    if (!slots_out || !allocation_out || n_layers == 0 || tile_bytes == 0) {\n        return false;\n    }\n    if (capacity != 0 && n_layers > std::numeric_limits<size_t>::max() / capacity) {\n        return false;\n    }\n    const size_t slots = n_layers * capacity;\n    if (slots != 0 && tile_bytes > std::numeric_limits<size_t>::max() / slots) {\n        return false;\n    }\n    const size_t payload = slots * tile_bytes;\n    if (payload > std::numeric_limits<size_t>::max() - (slha_tile_store::TILE_ALIGN - 1)) {\n        return false;\n    }\n    *slots_out = slots;\n    *allocation_out = payload + slha_tile_store::TILE_ALIGN - 1;\n    return true;\n}\n\nbool slha_resize_empty_tile_store(\n    slha_tile_store & store,\n    size_t n_layers,\n    size_t capacity,\n    size_t tile_bytes\n) {\n    std::lock_guard<std::mutex> lock(store.mutex);\n    if (store.n_layers == n_layers && store.tile_bytes == tile_bytes && store.capacity >= capacity) {\n        return true;\n    }\n    // Growing the store discards its index. It is legal only before any live\n    // tile exists; active-context growth must pass through the llama KV\n    // clear/reallocation seam instead of silently losing resident history.\n    if (std::any_of(store.valid.begin(), store.valid.end(), [](uint8_t value) { return value != 0; })) {\n        return false;\n    }\n\n    size_t slots = 0;\n    size_t allocation = 0;\n    if (!slha_checked_store_geometry(n_layers, capacity, tile_bytes, &slots, &allocation)) {\n        return false;\n    }\n    try {\n        std::vector<unsigned char> next_tiles(allocation, 0);\n        std::vector<uint8_t> next_valid(slots, 0);\n        const uintptr_t raw = reinterpret_cast<uintptr_t>(next_tiles.data());\n        const size_t pad = (slha_tile_store::TILE_ALIGN - (raw % slha_tile_store::TILE_ALIGN))\n            % slha_tile_store::TILE_ALIGN;\n        store.tiles.swap(next_tiles);\n        store.valid.swap(next_valid);\n        store.tile_base_offset = pad;\n        store.n_layers = n_layers;\n        store.capacity = capacity;\n        store.tile_bytes = tile_bytes;\n    } catch (const std::bad_alloc &) {\n        return false;\n    } catch (const std::length_error &) {\n        return false;\n    }\n    return true;\n}\n\n} // namespace\n\nbool slha_tile_store::init(size_t n_layers_, size_t capacity_, size_t tile_bytes_) {\n    std::lock_guard<std::mutex> lock(mutex);\n    if (std::any_of(valid.begin(), valid.end(), [](uint8_t value) { return value != 0; })) {\n        return false;\n    }\n    size_t slots = 0;\n    size_t allocation = 0;\n    if (!slha_checked_store_geometry(n_layers_, capacity_, tile_bytes_, &slots, &allocation)) {\n        return false;\n    }\n    try {\n        std::vector<unsigned char> next_tiles(allocation, 0);\n        std::vector<uint8_t> next_valid(slots, 0);\n        const uintptr_t raw = reinterpret_cast<uintptr_t>(next_tiles.data());\n        const size_t pad = (TILE_ALIGN - (raw % TILE_ALIGN)) % TILE_ALIGN;\n        tiles.swap(next_tiles);\n        valid.swap(next_valid);\n        tile_base_offset = pad;\n        n_layers = n_layers_;\n        capacity = capacity_;\n        tile_bytes = tile_bytes_;\n    } catch (const std::bad_alloc &) {\n        return false;\n    } catch (const std::length_error &) {\n        return false;\n    }\n    return true;\n}\n''',
    "llama checked dynamic tile-store geometry",
)

regex_once(
    cpp,
    r"const unsigned char\s*\*\s*slha_tile_store::read\(size_t layer, size_t position\) const \{.*?\n\}",
    '''const unsigned char * slha_tile_store::read(size_t layer, size_t position) const {\n    // Never expose mutable store storage after releasing `mutex`: a concurrent\n    // writer could otherwise race with a caller dereferencing the returned\n    // pointer. The public ABI is a 128-byte SLHA tile, so copy one immutable\n    // snapshot into thread-local, 128-aligned storage while still locked.\n    alignas(TILE_ALIGN) static thread_local unsigned char snapshot[128];\n    std::lock_guard<std::mutex> lock(mutex);\n    if (tile_bytes != sizeof(snapshot) || layer >= n_layers || position >= capacity) {\n        return nullptr;\n    }\n    const size_t index = layer * capacity + position;\n    if (index >= valid.size() || valid[index] == 0) {\n        return nullptr;\n    }\n    const size_t offset = tile_base_offset + index * tile_bytes;\n    if (offset > tiles.size() || tile_bytes > tiles.size() - offset) {\n        return nullptr;\n    }\n    std::memcpy(snapshot, tiles.data() + offset, tile_bytes);\n    return snapshot;\n}''',
    "llama snapshot tile-store reads under lock",
)

replace_once(
    cpp,
    '''    if (!k_tensor) {\n        return;\n    }\n\n    const int64_t original_bytes = n_tokens * head_dim * n_kv_heads * static_cast<int64_t>(ggml_type_size(k_tensor->type));\n    const int64_t reduced_bytes = n_tokens * 128; // 128 bytes per token (SLHAv2 tile size)\n\n    std::cout << "[SLHA] K-cache hook: GGML tensor keeps its real type/shape ("\n              << ggml_type_name(k_tensor->type) << ", " << original_bytes\n              << " bytes for " << n_tokens << " tokens x " << head_dim\n              << " x " << n_kv_heads << "); SLHA tiles live in the external "\n              << "tile store (" << reduced_bytes << " bytes, "\n              << (static_cast<double>(original_bytes) / static_cast<double>(std::max<int64_t>(reduced_bytes, 1)))\n              << "x projected reduction).\\n";\n''',
    '''    if (!k_tensor || n_tokens <= 0 || head_dim <= 0 || n_kv_heads <= 0) {\n        std::cerr << "[SLHA] invalid K-cache allocation metadata; refusing to size tile store\\n";\n        return;\n    }\n\n    auto & state = get_global_state();\n    size_t n_layers = 0;\n    slha_kv_mode kv_mode = SLHA_KV_LEGACY;\n    slha_score_mode score_mode = SLHA_SCORE_OFF;\n    {\n        std::lock_guard<std::mutex> state_lock(state.mutex);\n        n_layers = state.layers.size();\n        kv_mode = state.kv_mode;\n        score_mode = state.score_mode;\n    }\n    const size_t runtime_capacity = static_cast<size_t>(n_tokens);\n    const size_t tile_bytes = slha_tile_size();\n    if ((kv_mode == SLHA_KV_TILESTORE || score_mode != SLHA_SCORE_OFF) &&\n        !slha_resize_empty_tile_store(g_slha_tile_store, n_layers, runtime_capacity, tile_bytes)) {\n        std::cerr << "[SLHA] failed to size tile store from runtime K-cache capacity="\n                  << runtime_capacity << "; active data is never reallocated in place\\n";\n        return;\n    }\n\n    const size_t original_layer_bytes = ggml_nbytes(k_tensor);\n    if (runtime_capacity > std::numeric_limits<size_t>::max() / tile_bytes) {\n        std::cerr << "[SLHA] tile-store byte accounting overflow\\n";\n        return;\n    }\n    const size_t reduced_layer_bytes = runtime_capacity * tile_bytes;\n    const double reduction = reduced_layer_bytes == 0\n        ? 0.0\n        : static_cast<double>(original_layer_bytes) / static_cast<double>(reduced_layer_bytes);\n\n    std::cout << "[SLHA] K-cache hook: GGML tensor keeps its real type/shape ("\n              << ggml_type_name(k_tensor->type) << ", " << original_layer_bytes\n              << " bytes for runtime capacity " << runtime_capacity << " x head_dim "\n              << head_dim << " x kv_heads " << n_kv_heads << "); external SLHA "\n              << "capacity is derived from the runtime (" << reduced_layer_bytes\n              << " bytes/layer, " << reduction << "x projected per-layer reduction).\\n";\n''',
    "llama derive capacity and byte accounting from runtime metadata",
)

# ---------------------------------------------------------------------------
# CUDA: exact range checks before pointer arithmetic and same-context ownership
# for allocations, functions and streams.
# ---------------------------------------------------------------------------
cuda = Path("slhav2-vram/src/backends/cuda.rs")

replace_once(
    cuda,
    "700 is\n/// `CUDA_ERROR_INVALID_VALUE`",
    "700 is\n/// `CUDA_ERROR_ILLEGAL_ADDRESS`",
    "CUDA error-code documentation",
)

anchor = '''    pub fn backing_allocation_count() -> usize {\n        CUDA_BACKING_ALLOCATIONS.load(Ordering::SeqCst)\n    }\n\n'''
replace_once(
    cuda,
    anchor,
    anchor
    + '''    fn ensure_allocation_owner(\n        &self,\n        allocation: &CudaAllocation,\n        label: &str,\n    ) -> Result<(), CudaError> {\n        if Rc::ptr_eq(&self.inner, &allocation.owner) {\n            Ok(())\n        } else {\n            Err(CudaError(format!(\n                "{label}: allocation belongs to a different CUDA context"\n            )))\n        }\n    }\n\n    fn ensure_function_owner(\n        &self,\n        function: &CudaFunction,\n        label: &str,\n    ) -> Result<(), CudaError> {\n        if Rc::ptr_eq(&self.inner, &function.module.owner) {\n            Ok(())\n        } else {\n            Err(CudaError(format!(\n                "{label}: kernel belongs to a different CUDA context"\n            )))\n        }\n    }\n\n    fn ensure_stream_owner(&self, stream: &CudaStream, label: &str) -> Result<(), CudaError> {\n        if Rc::ptr_eq(&self.inner, &stream.owner) {\n            Ok(())\n        } else {\n            Err(CudaError(format!(\n                "{label}: stream belongs to a different CUDA context"\n            )))\n        }\n    }\n\n''',
    "CUDA context ownership helpers",
)

replace_once(
    cuda,
    '''    ) -> Result<(), CudaError> {\n        let end = dst_offset_bytes\n            .checked_add(src.len())\n''',
    '''    ) -> Result<(), CudaError> {\n        self.ensure_allocation_owner(dst, "copy_to_device_at")?;\n        let end = dst_offset_bytes\n            .checked_add(src.len())\n''',
    "CUDA sync H2D ownership",
)
replace_once(
    cuda,
    '''    ) -> Result<(), CudaError> {\n        let end = src_offset_bytes\n            .checked_add(dst.len())\n''',
    '''    ) -> Result<(), CudaError> {\n        self.ensure_allocation_owner(src, "copy_to_host_at")?;\n        let end = src_offset_bytes\n            .checked_add(dst.len())\n''',
    "CUDA sync D2H ownership",
)

replace_once(
    cuda,
    '''        let q_coarse_ptr = q_coarse_dev.ptr + q_coarse_offset as u64;\n        let q_sign_ptr = q_sign_dev.ptr + q_sign_offset as u64;\n        let tiles_ptr = tiles_dev.ptr + tiles_offset as u64;\n        let scores_ptr = scores_dev.ptr + scores_offset as u64;\n\n        let grid_dim = (num_tiles as usize).div_ceil(256) as u32;\n''',
    '''        self.ensure_allocation_owner(q_coarse_dev, "score_tiles_at q_coarse")?;\n        self.ensure_allocation_owner(q_sign_dev, "score_tiles_at q_sign")?;\n        self.ensure_allocation_owner(tiles_dev, "score_tiles_at tiles")?;\n        self.ensure_allocation_owner(scores_dev, "score_tiles_at scores")?;\n        self.ensure_function_owner(kernel, "score_tiles_at")?;\n\n        let n = num_tiles as usize;\n        let q_coarse_bytes = crate::codec::D_C * core::mem::size_of::<f32>();\n        let q_sign_bytes = crate::codec::RESIDUAL_WORDS * core::mem::size_of::<u64>();\n        let tile_bytes = n\n            .checked_mul(crate::codec::TILE_BYTES)\n            .ok_or_else(|| CudaError("score_tiles_at: tile byte count overflow".into()))?;\n        let score_bytes = n\n            .checked_mul(core::mem::size_of::<f32>())\n            .ok_or_else(|| CudaError("score_tiles_at: score byte count overflow".into()))?;\n        let check_range = |label: &str, offset: usize, need: usize, len: usize| {\n            let end = offset\n                .checked_add(need)\n                .ok_or_else(|| CudaError(format!("score_tiles_at: {label} range overflow")))?;\n            if end > len {\n                return Err(CudaError(format!(\n                    "score_tiles_at: {label} offset {offset} + size {need} exceeds allocation size {len}"\n                )));\n            }\n            Ok(())\n        };\n        check_range("q_coarse", q_coarse_offset, q_coarse_bytes, q_coarse_dev.len)?;\n        check_range("q_sign", q_sign_offset, q_sign_bytes, q_sign_dev.len)?;\n        check_range("tiles", tiles_offset, tile_bytes, tiles_dev.len)?;\n        check_range("scores", scores_offset, score_bytes, scores_dev.len)?;\n\n        let q_coarse_ptr = q_coarse_dev.ptr + q_coarse_offset as u64;\n        let q_sign_ptr = q_sign_dev.ptr + q_sign_offset as u64;\n        let tiles_ptr = tiles_dev.ptr + tiles_offset as u64;\n        let scores_ptr = scores_dev.ptr + scores_offset as u64;\n\n        let grid_dim = n.div_ceil(256) as u32;\n''',
    "CUDA offset/range/context validation",
)

regex_once(
    cuda,
    r'''(    pub fn score_tiles\(.*?        if num_tiles <= 0 \{\n            return Ok\(\(\)\);\n        \}\n)(        let n = num_tiles as usize;)''',
    r'''\1        self.ensure_allocation_owner(q_coarse_dev, "score_tiles q_coarse")?;\n        self.ensure_allocation_owner(q_sign_dev, "score_tiles q_sign")?;\n        self.ensure_allocation_owner(tiles_dev, "score_tiles tiles")?;\n        self.ensure_allocation_owner(scores_dev, "score_tiles scores")?;\n        self.ensure_function_owner(kernel, "score_tiles")?;\n\2''',
    "CUDA score_tiles same-context validation",
)

regex_once(
    cuda,
    r'''(    pub fn score_tiles_on_stream\(.*?        if num_tiles <= 0 \{\n            return Ok\(\(\)\);\n        \}\n)(        let n = num_tiles as usize;)''',
    r'''\1        self.ensure_allocation_owner(q_coarse_dev, "score_tiles_on_stream q_coarse")?;\n        self.ensure_allocation_owner(q_sign_dev, "score_tiles_on_stream q_sign")?;\n        self.ensure_allocation_owner(tiles_dev, "score_tiles_on_stream tiles")?;\n        self.ensure_allocation_owner(scores_dev, "score_tiles_on_stream scores")?;\n        self.ensure_function_owner(kernel, "score_tiles_on_stream")?;\n        self.ensure_stream_owner(stream, "score_tiles_on_stream")?;\n\2''',
    "CUDA stream score same-context validation",
)

replace_once(
    cuda,
    '''    ) -> Result<(), CudaError> {\n        let end = dst_offset\n            .checked_add(src.len())\n            .ok_or_else(|| CudaError("copy_to_device_async: offset + length overflow".into()))?;\n''',
    '''    ) -> Result<(), CudaError> {\n        self.ensure_allocation_owner(dst, "copy_to_device_async")?;\n        self.ensure_stream_owner(stream, "copy_to_device_async")?;\n        let end = dst_offset\n            .checked_add(src.len())\n            .ok_or_else(|| CudaError("copy_to_device_async: offset + length overflow".into()))?;\n''',
    "CUDA async H2D ownership",
)
replace_once(
    cuda,
    '''    ) -> Result<(), CudaError> {\n        let end = src_offset\n            .checked_add(dst.len())\n            .ok_or_else(|| CudaError("copy_to_host_async: offset + length overflow".into()))?;\n''',
    '''    ) -> Result<(), CudaError> {\n        self.ensure_allocation_owner(src, "copy_to_host_async")?;\n        self.ensure_stream_owner(stream, "copy_to_host_async")?;\n        let end = src_offset\n            .checked_add(dst.len())\n            .ok_or_else(|| CudaError("copy_to_host_async: offset + length overflow".into()))?;\n''',
    "CUDA async D2H ownership",
)

replace_once(
    cuda,
    '''    ) -> Result<(), CudaError> {\n        if dst_offset\n            .checked_add(src.len())\n            .is_none_or(|end| end > dst.len)\n''',
    '''    ) -> Result<(), CudaError> {\n        self.ensure_allocation_owner(dst, "copy_to_device")?;\n        if dst_offset\n            .checked_add(src.len())\n            .is_none_or(|end| end > dst.len)\n''',
    "CUDA DeviceEngine H2D ownership",
)
replace_once(
    cuda,
    '''    ) -> Result<(), CudaError> {\n        if src_offset\n            .checked_add(dst.len())\n            .is_none_or(|end| end > src.len)\n''',
    '''    ) -> Result<(), CudaError> {\n        self.ensure_allocation_owner(src, "copy_to_host")?;\n        if src_offset\n            .checked_add(dst.len())\n            .is_none_or(|end| end > src.len)\n''',
    "CUDA DeviceEngine D2H ownership",
)

# Small source fixes kept in the same atomic patch so the following CI run is
# known to compile the intended audited semantics.
tiers = Path("elastic/elastic-core/src/tiers.rs")
replace_once(
    tiers,
    "#[derive(Clone, Debug)]\npub struct TierMachine {",
    "#[derive(Clone, Debug, PartialEq, Eq)]\npub struct TierMachine {",
    "TierMachine equality for Result assertions",
)

controller = Path("elastic/elastic-core/src/controller.rs")
replace_once(
    controller,
    "use crate::decision::{Candidate, DecisionTrace, Rejection};",
    "use crate::decision::{Candidate, DecisionTrace};",
    "controller remove stale Rejection import",
)
replace_once(
    controller,
    '''        trace.consider(\n            Candidate::new("none", 0.0, 0.0, 0.0, 0.0),\n            Some(Rejection::CostTooHigh),\n        );\n''',
    '''        trace.consider(Candidate::new("none", 0.0, 0.0, 0.0, 0.0), None);\n''',
    "controller truthful no-action trace",
)

pipeline = Path("slhav2-vram/src/pipeline.rs")
replace_once(
    pipeline,
    "pub fn score_tiles_cpu<E: DeviceEngine>(mut input: ScoringInput<'_, E>) {",
    "pub fn score_tiles_cpu<E: DeviceEngine>(input: ScoringInput<'_, E>) {",
    "pipeline remove unnecessary mutability",
)

print("all P0 source transformations applied")
