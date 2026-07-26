use std::collections::HashMap;
use std::sync::Mutex;

use crate::traits::{DeviceEngine, DevicePointer, VramError, VramResult};

/// CPU reference backend — ground truth for numerical validation.
pub struct CpuRefBackend {
    name: &'static str,
    heap: Mutex<HeapState>,
}

struct HeapState {
    next_id: u64,
    blocks: HashMap<u64, Vec<u8>>,
}

impl CpuRefBackend {
    pub fn new(_total_ram_hint_mb: usize) -> Self {
        CpuRefBackend {
            name: "cpu-ref",
            heap: Mutex::new(HeapState {
                next_id: 1,
                blocks: HashMap::new(),
            }),
        }
    }
}

impl DeviceEngine for CpuRefBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn allocate(&self, size_bytes: usize) -> VramResult<DevicePointer> {
        let mut state = self.heap.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.blocks.insert(id, vec![0u8; size_bytes]);
        Ok(DevicePointer { raw: id, size: size_bytes })
    }

    fn free(&self, ptr: DevicePointer) -> VramResult<()> {
        let mut state = self.heap.lock().unwrap();
        state.blocks.remove(&ptr.raw).ok_or_else(|| {
            VramError::InvalidPointer(format!("CPU heap block {} not found", ptr.raw))
        })?;
        Ok(())
    }

    fn copy_to_device(&self, src: &[u8], dst: &DevicePointer) -> VramResult<()> {
        let mut state = self.heap.lock().unwrap();
        let block = state.blocks.get_mut(&dst.raw).ok_or_else(|| {
            VramError::InvalidPointer(format!("CPU heap block {} not found", dst.raw))
        })?;
        if src.len() > block.len() {
            return Err(VramError::CopyToDeviceFailed(format!(
                "source {} bytes exceeds destination {} bytes", src.len(), block.len()
            )));
        }
        block[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_to_host(&self, src: &DevicePointer, dst: &mut [u8]) -> VramResult<()> {
        let state = self.heap.lock().unwrap();
        let block = state.blocks.get(&src.raw).ok_or_else(|| {
            VramError::InvalidPointer(format!("CPU heap block {} not found", src.raw))
        })?;
        if dst.len() > block.len() {
            return Err(VramError::CopyToHostFailed(format!(
                "destination {} bytes exceeds source {} bytes", dst.len(), block.len()
            )));
        }
        dst.copy_from_slice(&block[..dst.len()]);
        Ok(())
    }

    fn synchronize(&self) -> VramResult<()> {
        Ok(())
    }

    fn launch_lowrank_matmul(
        &self,
        input: &DevicePointer,
        weight_lowrank: &DevicePointer,
        output: &DevicePointer,
        dim_m: usize,
        dim_n: usize,
        dim_k: usize,
    ) -> VramResult<()> {
        let group_size: usize = 16;
        let num_groups_per_row = dim_k / group_size;
        let scale_offset_bytes = dim_n * (dim_k / 2);

        // Read input f32 data
        let input_data: Vec<f32> = {
            let state = self.heap.lock().unwrap();
            let block = state.blocks.get(&input.raw).ok_or_else(|| {
                VramError::InvalidPointer(format!("input block {} not found", input.raw))
            })?;
            block
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        // Read weights and scales
        let (weight_q, scales): (Vec<u8>, Vec<f32>) = {
            let state = self.heap.lock().unwrap();
            let block = state.blocks.get(&weight_lowrank.raw).ok_or_else(|| {
                VramError::InvalidPointer(format!("weight block {} not found", weight_lowrank.raw))
            })?;
            let w = block[..scale_offset_bytes].to_vec();
            let s: Vec<f32> = block[scale_offset_bytes..]
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (w, s)
        };

        // Compute Y = W * X with on-the-fly INT4 dequant
        let mut output_data = vec![0.0f32; dim_m * dim_n];
        for m in 0..dim_m {
            for n in 0..dim_n {
                let mut sum = 0.0f32;
                for k in 0..dim_k {
                    let packed_idx = n * (dim_k / 2) + k / 2;
                    let packed = weight_q[packed_idx];
                    let nibble = if k % 2 == 0 { packed & 0x0F } else { (packed >> 4) & 0x0F };
                    let signed_val = if nibble & 0x08 != 0 {
                        (nibble as i8) - 16
                    } else {
                        nibble as i8
                    } as f32;
                    let group = k / group_size;
                    let scale = scales[n * num_groups_per_row + group];
                    sum += input_data[m * dim_k + k] * signed_val * scale;
                }
                output_data[m * dim_n + n] = sum;
            }
        }

        // Write output
        {
            let mut state = self.heap.lock().unwrap();
            let block = state.blocks.get_mut(&output.raw).ok_or_else(|| {
                VramError::InvalidPointer(format!("output block {} not found", output.raw))
            })?;
            let out_bytes: Vec<u8> = output_data
                .iter()
                .flat_map(|&v| v.to_ne_bytes())
                .collect();
            block[..out_bytes.len()].copy_from_slice(&out_bytes);
        }

        Ok(())
    }

    fn memory_info(&self) -> VramResult<(usize, usize)> {
        let state = self.heap.lock().unwrap();
        let used: usize = state.blocks.values().map(|b| b.len()).sum();
        Ok((usize::MAX, usize::MAX - used))
    }
}
