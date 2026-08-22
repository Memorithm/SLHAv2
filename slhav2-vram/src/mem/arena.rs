use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

/// A live sub-allocation in a [`DeviceArena`].
pub struct ArenaSlice {
    pub offset: u64,
    pub size: u64,
    pub cookie: u64,
}

impl ArenaSlice {
    pub fn is_valid(&self) -> bool {
        self.cookie != 0
    }
}

/// Natural alignment of a serialized SLHAv2 tile.
const ALIGN: u64 = 128;

fn align_up(offset: u64) -> Option<u64> {
    offset.checked_add(ALIGN - 1).map(|v| v & !(ALIGN - 1))
}

#[derive(Clone, Copy, Debug)]
struct FreeBlock {
    size: u64,
}

#[derive(Clone, Copy, Debug)]
struct AllocBlock {
    offset: u64,
    size: u64,
}

/// Deterministic sub-allocator over one device backing allocation.
///
/// Every allocation starts at a 128-byte-aligned offset. Alignment prefixes
/// and suffixes are kept in the free map instead of being silently lost, so
/// the exact invariant is always:
///
/// `used_bytes + free_bytes == capacity`.
pub struct DeviceArena<A> {
    backing: A,
    capacity: u64,
    /// Offset -> free block.
    free: BTreeMap<u64, FreeBlock>,
    alloc_map: HashMap<u64, AllocBlock>,
    used: u64,
    next_cookie: AtomicU64,
}

impl<A> DeviceArena<A> {
    pub fn new(backing: A, capacity: u64) -> Self {
        let mut arena = Self {
            backing,
            capacity,
            free: BTreeMap::new(),
            alloc_map: HashMap::new(),
            used: 0,
            next_cookie: AtomicU64::new(1),
        };
        if capacity != 0 {
            arena.free.insert(0, FreeBlock { size: capacity });
        }
        arena
    }

    pub fn backing(&self) -> &A {
        &self.backing
    }

    pub fn backing_mut(&mut self) -> &mut A {
        &mut self.backing
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Bytes represented by actual free-list blocks.
    pub fn free_bytes(&self) -> u64 {
        self.free.values().map(|b| b.size).sum()
    }

    /// Alignment no longer consumes bytes permanently. Kept for API
    /// compatibility; a correct arena always reports zero overhead.
    pub fn overhead_bytes(&self) -> u64 {
        0
    }

    /// Conservation invariant for the backing allocation.
    pub fn accounting_is_conserved(&self) -> bool {
        self.used
            .checked_add(self.free_bytes())
            .is_some_and(|v| v == self.capacity)
    }

    pub fn live_allocations(&self) -> usize {
        self.alloc_map.len()
    }

    /// Allocate `size` bytes at a 128-byte-aligned offset.
    ///
    /// The search tests the *post-alignment* usable extent of each free block;
    /// a misaligned block that cannot fit the request is skipped rather than
    /// causing a false allocation failure.
    pub fn allocate(&mut self, size: u64) -> Option<ArenaSlice> {
        if size == 0 || size > self.capacity {
            return None;
        }

        let fit = self.free.iter().find_map(|(&offset, block)| {
            let aligned = align_up(offset)?;
            let end = offset.checked_add(block.size)?;
            let alloc_end = aligned.checked_add(size)?;
            (alloc_end <= end).then_some((offset, block.size, aligned, alloc_end, end))
        });
        let (offset, block_size, aligned, alloc_end, block_end) = fit?;
        self.free.remove(&offset);

        // Preserve the alignment prefix as free memory. It may be too small
        // for the current request but can coalesce with neighbours later.
        if aligned > offset {
            self.insert_free(offset, aligned - offset);
        }
        if alloc_end < block_end {
            self.insert_free(alloc_end, block_end - alloc_end);
        }

        debug_assert_eq!(block_size, block_end - offset);

        let mut cookie = self.next_cookie.fetch_add(1, Ordering::Relaxed);
        // Cookie zero is reserved for invalid slices; handle wrap defensively.
        if cookie == 0 {
            cookie = self.next_cookie.fetch_add(1, Ordering::Relaxed);
            if cookie == 0 {
                return None;
            }
        }

        self.alloc_map.insert(
            cookie,
            AllocBlock {
                offset: aligned,
                size,
            },
        );
        self.used = self.used.checked_add(size)?;
        debug_assert!(self.accounting_is_conserved());

        Some(ArenaSlice {
            offset: aligned,
            size,
            cookie,
        })
    }

    /// Free a live slice. Invalid/stale cookies are ignored.
    pub fn free(&mut self, slice: &ArenaSlice) {
        if slice.cookie == 0 {
            return;
        }
        let Some(block) = self.alloc_map.remove(&slice.cookie) else {
            return;
        };
        self.used = self.used.saturating_sub(block.size);
        self.insert_free(block.offset, block.size);
        debug_assert!(self.accounting_is_conserved());
    }

    /// Insert and coalesce with immediately adjacent free blocks.
    fn insert_free(&mut self, offset: u64, size: u64) {
        if size == 0 {
            return;
        }
        let mut off = offset;
        let mut sz = size;

        if let Some((&prev_off, prev)) = self.free.range(..off).next_back() {
            if prev_off.checked_add(prev.size) == Some(off) {
                let prev_size = prev.size;
                self.free.remove(&prev_off);
                off = prev_off;
                sz = sz.saturating_add(prev_size);
            }
        }

        if let Some((&next_off, next)) = self.free.range(off.saturating_add(1)..).next() {
            if off.checked_add(sz) == Some(next_off) {
                let next_size = next.size;
                self.free.remove(&next_off);
                sz = sz.saturating_add(next_size);
            }
        }

        self.free.insert(off, FreeBlock { size: sz });
    }

    pub fn reset(&mut self) {
        self.free.clear();
        self.alloc_map.clear();
        self.used = 0;
        if self.capacity != 0 {
            self.free.insert(
                0,
                FreeBlock {
                    size: self.capacity,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_aligned_allocation() {
        let mut arena = DeviceArena::new(0u64, 1024);
        let a = arena.allocate(128).unwrap();
        let b = arena.allocate(128).unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 128);
        assert_eq!(arena.used_bytes(), 256);
        assert_eq!(arena.free_bytes(), 768);
        assert!(arena.accounting_is_conserved());
    }

    #[test]
    fn zero_and_oversized_requests_are_rejected() {
        let mut arena = DeviceArena::new(0u64, 256);
        assert!(arena.allocate(0).is_none());
        assert!(arena.allocate(257).is_none());
        assert!(arena.accounting_is_conserved());
    }

    #[test]
    fn alignment_prefix_is_not_lost() {
        let mut arena = DeviceArena::new(0u64, 1024);
        let a = arena.allocate(100).unwrap();
        let b = arena.allocate(100).unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 128);
        // [100,128) remains represented in the free map.
        assert_eq!(arena.used_bytes(), 200);
        assert_eq!(arena.free_bytes(), 824);
        assert_eq!(arena.overhead_bytes(), 0);
        assert!(arena.accounting_is_conserved());

        arena.free(&a);
        arena.free(&b);
        assert_eq!(arena.free_bytes(), 1024);
        assert!(arena.accounting_is_conserved());
    }

    #[test]
    fn search_skips_block_that_fails_after_alignment() {
        let mut arena = DeviceArena::new(0u64, 1024);
        let a = arena.allocate(100).unwrap();
        let b = arena.allocate(128).unwrap();
        let c = arena.allocate(128).unwrap();
        arena.free(&b);
        // The small prefix [100,128) appears before b's free block and cannot
        // satisfy 128 after alignment; allocation must continue searching.
        let d = arena.allocate(128).unwrap();
        assert_eq!(d.offset, 128);
        arena.free(&a);
        arena.free(&c);
        arena.free(&d);
        assert_eq!(arena.free_bytes(), arena.capacity());
    }

    #[test]
    fn stale_cookie_is_a_noop() {
        let mut arena = DeviceArena::new(0u64, 1024);
        let a = arena.allocate(128).unwrap();
        arena.free(&a);
        let before = arena.free_bytes();
        arena.free(&a);
        assert_eq!(arena.free_bytes(), before);
        assert!(arena.accounting_is_conserved());
    }

    #[test]
    fn repeated_grow_shrink_conserves_every_byte() {
        let mut arena = DeviceArena::new(0u64, 1 << 20);
        for round in 0..100 {
            let mut live = Vec::new();
            for i in 0..200 {
                let size = 1 + ((round * 17 + i * 31) % 300) as u64;
                if let Some(s) = arena.allocate(size) {
                    assert_eq!(s.offset % ALIGN, 0);
                    live.push(s);
                }
                assert!(arena.accounting_is_conserved());
            }
            for s in live.iter().rev() {
                arena.free(s);
                assert!(arena.accounting_is_conserved());
            }
            assert_eq!(arena.used_bytes(), 0);
            assert_eq!(arena.free_bytes(), arena.capacity());
        }
    }

    #[test]
    fn fragmentation_is_reported_not_hidden_as_overhead() {
        let mut arena = DeviceArena::new(0u64, 4096);
        let mut slices = Vec::new();
        for _ in 0..20 {
            slices.push(arena.allocate(128).unwrap());
        }
        for (i, s) in slices.iter().enumerate() {
            if i % 2 == 0 {
                arena.free(s);
            }
        }
        assert!(arena.free_bytes() > 0);
        assert_eq!(arena.overhead_bytes(), 0);
        assert!(arena.accounting_is_conserved());
    }

    #[test]
    fn reset_restores_full_capacity() {
        let mut arena = DeviceArena::new(0u64, 1024);
        arena.allocate(100).unwrap();
        arena.allocate(100).unwrap();
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.free_bytes(), 1024);
        assert!(arena.accounting_is_conserved());
    }
}
