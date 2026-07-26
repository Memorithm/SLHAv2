use std::error::Error;
use std::fmt;

use crate::traits::{DeviceAllocation, DeviceEngine};

#[derive(Clone)]
pub struct CpuAllocation {
    data: Vec<u8>,
}

impl CpuAllocation {
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl DeviceAllocation for CpuAllocation {
    fn size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone, Default)]
pub struct CpuEngine;

impl CpuEngine {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct CpuError(String);

impl fmt::Display for CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CpuError: {}", self.0)
    }
}

impl Error for CpuError {}

impl DeviceEngine for CpuEngine {
    type Alloc = CpuAllocation;
    type Error = CpuError;

    fn allocate(&self, size: usize) -> Result<CpuAllocation, CpuError> {
        Ok(CpuAllocation {
            data: vec![0u8; size],
        })
    }

    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &mut CpuAllocation,
        dst_offset: usize,
    ) -> Result<(), CpuError> {
        if dst_offset + src.len() > dst.data.len() {
            return Err(CpuError(format!(
                "copy_to_device: offset {}+{} exceeds allocation size {}",
                dst_offset,
                src.len(),
                dst.data.len()
            )));
        }
        dst.data[dst_offset..dst_offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_to_host(
        &self,
        src: &CpuAllocation,
        src_offset: usize,
        dst: &mut [u8],
    ) -> Result<(), CpuError> {
        if src_offset + dst.len() > src.data.len() {
            return Err(CpuError(format!(
                "copy_to_host: offset {}+{} exceeds allocation size {}",
                src_offset,
                dst.len(),
                src.data.len()
            )));
        }
        dst.copy_from_slice(&src.data[src_offset..src_offset + dst.len()]);
        Ok(())
    }

    fn set_device(&self) -> Result<(), CpuError> {
        Ok(())
    }

    fn synchronize(&self) -> Result<(), CpuError> {
        Ok(())
    }
}
