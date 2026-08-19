use std::error::Error;
use std::fmt;

use crate::traits::{DeviceAllocation, DeviceEngine};

/// Host-backed allocation used by the CPU reference backend.
#[derive(Clone)]
pub struct CpuAllocation {
    data: Vec<u8>,
}

impl CpuAllocation {
    /// Borrow allocation bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutably borrow allocation bytes.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl DeviceAllocation for CpuAllocation {
    fn size(&self) -> usize {
        self.data.len()
    }
}

/// CPU reference device engine.
#[derive(Clone, Default)]
pub struct CpuEngine;

impl CpuEngine {
    /// Construct a CPU engine.
    pub fn new() -> Self {
        Self
    }
}

/// CPU backend error.
#[derive(Debug, Clone, PartialEq, Eq)]
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
        let end = dst_offset.checked_add(src.len()).ok_or_else(|| {
            CpuError("copy_to_device: offset + length overflow".to_string())
        })?;
        if end > dst.data.len() {
            return Err(CpuError(format!(
                "copy_to_device: offset {} + size {} exceeds allocation size {}",
                dst_offset,
                src.len(),
                dst.data.len()
            )));
        }
        dst.data[dst_offset..end].copy_from_slice(src);
        Ok(())
    }

    fn copy_to_host(
        &self,
        src: &CpuAllocation,
        src_offset: usize,
        dst: &mut [u8],
    ) -> Result<(), CpuError> {
        let end = src_offset.checked_add(dst.len()).ok_or_else(|| {
            CpuError("copy_to_host: offset + length overflow".to_string())
        })?;
        if end > src.data.len() {
            return Err(CpuError(format!(
                "copy_to_host: offset {} + size {} exceeds allocation size {}",
                src_offset,
                dst.len(),
                src.data.len()
            )));
        }
        dst.copy_from_slice(&src.data[src_offset..end]);
        Ok(())
    }

    fn set_device(&self) -> Result<(), CpuError> {
        Ok(())
    }

    fn synchronize(&self) -> Result<(), CpuError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_bounds_reject_overflow_and_oob() {
        let engine = CpuEngine::new();
        let mut allocation = engine.allocate(16).unwrap();
        assert!(engine
            .copy_to_device(&[1, 2, 3], &mut allocation, usize::MAX)
            .is_err());
        assert!(engine
            .copy_to_device(&[1, 2, 3], &mut allocation, 15)
            .is_err());

        let mut dst = [0u8; 4];
        assert!(engine
            .copy_to_host(&allocation, usize::MAX, &mut dst)
            .is_err());
        assert!(engine.copy_to_host(&allocation, 13, &mut dst).is_err());
    }

    #[test]
    fn valid_copy_roundtrips() {
        let engine = CpuEngine::new();
        let mut allocation = engine.allocate(8).unwrap();
        engine
            .copy_to_device(&[1, 2, 3, 4], &mut allocation, 2)
            .unwrap();
        let mut dst = [0u8; 4];
        engine.copy_to_host(&allocation, 2, &mut dst).unwrap();
        assert_eq!(dst, [1, 2, 3, 4]);
    }
}
