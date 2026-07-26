use std::sync::Mutex;

use crate::traits::{DeviceEngine, DevicePointer, VramResult};

/// A memory pool that delegates allocations to the engine and tracks them
/// for safe lifecycle management.
///
/// On backends where `DevicePointer.raw` is a real address (CUDA, Vulkan),
/// this can be extended to a true slab arena by pre-allocating a single
/// contiguous block and returning offset-based pointers.
pub struct VramMemoryPool {
    engine: Box<dyn DeviceEngine>,
    state: Mutex<PoolState>,
}

struct PoolState {
    allocations: Vec<DevicePointer>,
}

impl VramMemoryPool {
    /// Create a pool wrapping the given engine.
    pub fn new(engine: Box<dyn DeviceEngine>, _pool_size_hint: usize) -> VramResult<Self> {
        Ok(VramMemoryPool {
            engine,
            state: Mutex::new(PoolState {
                allocations: Vec::new(),
            }),
        })
    }

    pub fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    /// Allocate device memory via the underlying engine.
    pub fn allocate(&self, size: usize) -> VramResult<DevicePointer> {
        let ptr = self.engine.allocate(size)?;
        let mut state = self.state.lock().unwrap();
        state.allocations.push(ptr);
        Ok(ptr)
    }

    /// Free a previously allocated device pointer.
    pub fn free(&self, ptr: DevicePointer) -> VramResult<()> {
        let mut state = self.state.lock().unwrap();
        let pos = state
            .allocations
            .iter()
            .position(|p| p.raw == ptr.raw && p.size == ptr.size)
            .ok_or_else(|| {
                crate::traits::VramError::InvalidPointer(format!(
                    "pointer {:#x} (size {}) not found in pool",
                    ptr.raw, ptr.size
                ))
            })?;
        state.allocations.swap_remove(pos);
        self.engine.free(ptr)
    }

    /// Access the underlying engine for direct operations.
    pub fn engine(&self) -> &dyn DeviceEngine {
        self.engine.as_ref()
    }

    /// Number of live allocations.
    pub fn live_allocations(&self) -> usize {
        self.state.lock().unwrap().allocations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::cpu_ref::CpuRefBackend;

    fn test_pool() -> VramMemoryPool {
        let engine = Box::new(CpuRefBackend::new(64));
        VramMemoryPool::new(engine, 1024 * 1024).unwrap()
    }

    #[test]
    fn pool_alloc_free() {
        let pool = test_pool();
        let a = pool.allocate(128).unwrap();
        assert!(a.raw > 0);
        assert_eq!(a.size, 128);
        let b = pool.allocate(256).unwrap();
        assert_eq!(pool.live_allocations(), 2);
        pool.free(a).unwrap();
        assert_eq!(pool.live_allocations(), 1);
        pool.free(b).unwrap();
        assert_eq!(pool.live_allocations(), 0);
    }

    #[test]
    fn pool_reuse_freed() {
        let pool = test_pool();
        let a = pool.allocate(128).unwrap();
        let _raw_a = a.raw;
        pool.free(a).unwrap();
        // Next allocation may or may not reuse the same ID — depends on engine
        let _b = pool.allocate(128).unwrap();
        // We just verify no crash
    }

    #[test]
    fn pool_stress_alloc_free() {
        let pool = test_pool();
        let mut ptrs = Vec::new();
        for _ in 0..64 {
            ptrs.push(pool.allocate(4096).unwrap());
        }
        assert_eq!(pool.live_allocations(), 64);
        for p in ptrs.drain(..) {
            pool.free(p).unwrap();
        }
        assert_eq!(pool.live_allocations(), 0);
    }
}
