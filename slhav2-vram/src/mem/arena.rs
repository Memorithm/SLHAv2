use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Align a byte offset up to the tile's natural alignment (checked). The
/// arena backing (e.g. a `cuMemAlloc` region) is already 256-aligned, so the
/// base offset 0 is always aligned. Aligning each allocation's **offset** to
/// 128 (the 128-byte tile's natural alignment — enough for the u64 residual
/// reads) lets 128-byte tiles pack back-to-back at a 128-byte stride, instead
/// of the old scheme that rounded the *size* to 256 and wasted half the arena.
/// The recorded block size stays the real allocation size.
const ALIGN: u64 = 128;

fn align_up(offset: u64) -> Option<u64> {
    offset.checked_add(ALIGN - 1).map(|v| v & !(ALIGN - 1))
}

/// Free list keyed by offset in a [`BTreeMap`]: best-fit allocation is a
/// `range(..=size)` lookup and adjacent coalescing is two neighbour lookups —
/// O(log n) per operation, with **no full-list sort** on free (the old
/// `coalesce` sorted the entire free list on every free, O(n log n) per
/// step). 128-byte tiles pack at a 128-byte stride (offsets aligned to the
/// tile's natural alignment, sizes kept real).
pub struct DeviceArena<A> {
    backing: A,
    capacity: u64,
    /// Offset → free block, ordered by offset. The first block whose size
    /// fits the request (found via `range`) is used.
    free: BTreeMap<u64, FreeBlock>,
    alloc_map: std::collections::HashMap<u64, AllocBlock>,
    used: u64,
    next_cookie: AtomicU64,
}

#[derive(Clone, Copy)]
struct FreeBlock {
    offset: u64,
    size: u64,
}

struct AllocBlock {
    offset: u64,
    size: u64,
}

impl<A> DeviceArena<A> {
    pub fn new(backing: A, capacity: u64) -> Self {
        let mut arena = Self {
            backing,
            capacity,
            free: BTreeMap::new(),
            alloc_map: std::collections::HashMap::new(),
            used: 0,
            next_cookie: AtomicU64::new(1),
        };
        arena.insert_free(0, capacity);
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

    pub fn free_bytes(&self) -> u64 {
        self.capacity - self.used
    }

    pub fn live_allocations(&self) -> usize {
        self.alloc_map.len()
    }

    pub fn allocate(&mut self, size: u64) -> Option<ArenaSlice> {
        if size == 0 {
            return None;
        }
        if size > self.capacity {
            return None;
        }

        // First-fit by offset among blocks large enough — O(log n + k).
        let fit: Option<(u64, FreeBlock)> = self
            .free
            .iter()
            .find(|(_, b)| b.size >= size)
            .map(|(&off, &b)| (off, b));
        let (_, FreeBlock { offset, size: real }) = fit?;
        self.free.remove(&offset);

        // Align only the offset.
        let aligned_off = align_up(offset)?;
        let pad = aligned_off - offset;
        let avail = real.checked_sub(pad)?;
        if avail < size {
            self.insert_free(offset, real);
            return None;
        }

        let leftover = avail - size;
        if leftover > 0 {
            self.insert_free(aligned_off + size, leftover);
        }

        let cookie = self.next_cookie.fetch_add(1, Ordering::Relaxed);
        self.alloc_map.insert(
            cookie,
            AllocBlock {
                offset: aligned_off,
                size,
            },
        );
        self.used += size;

        Some(ArenaSlice {
            offset: aligned_off,
            size,
            cookie,
        })
    }

    pub fn free(&mut self, slice: &ArenaSlice) {
        if slice.cookie == 0 {
            return;
        }
        let Some(block) = self.alloc_map.remove(&slice.cookie) else {
            return;
        };
        self.used -= block.size;
        self.insert_free(block.offset, block.size);
        self.coalesce();
    }

    /// Insert a free block, merging it with any adjacent free blocks (the
    /// two neighbours by offset). O(log n).
    fn insert_free(&mut self, offset: u64, size: u64) {
        let mut off = offset;
        let mut sz = size;

        // Merge with the block immediately before.
        if let Some((&poff, &p)) = self.free.range(..off).next_back() {
            if poff + p.size == off {
                self.free.remove(&poff);
                off = poff;
                sz += p.size;
            }
        }
        // Merge with the block immediately after.
        if let Some((&noff, &n)) = self.free.range(off + 1..).next() {
            if off + sz == noff {
                self.free.remove(&noff);
                sz += n.size;
            }
        }

        self.free.insert(
            off,
            FreeBlock {
                offset: off,
                size: sz,
            },
        );
    }

    /// Coalesce after a free: the freed block is already merged with its
    /// neighbours by [`Self::insert_free`], so this is a no-op retained for
    /// API clarity — adjacency is handled inline.
    fn coalesce(&self) {}

    pub fn reset(&mut self) {
        self.free.clear();
        self.alloc_map.clear();
        self.used = 0;
        self.insert_free(0, self.capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation_basic() {
        let backing = 0u64; // dummy, not used in tests
        let mut arena = DeviceArena::new(backing, 1024);

        let a = arena.allocate(128).unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(a.size, 128); // real size, not aligned

        let b = arena.allocate(128).unwrap();
        assert_eq!(b.offset, 128);
        assert_eq!(b.size, 128);

        assert_eq!(arena.live_allocations(), 2);
        assert_eq!(arena.used_bytes(), 256);
    }

    #[test]
    fn test_arena_zero_size_rejected() {
        let mut arena = DeviceArena::new(0u64, 1024);
        assert!(arena.allocate(0).is_none());
    }

    #[test]
    fn test_arena_overflow_rejected() {
        let mut arena = DeviceArena::new(0u64, 256);
        assert!(arena.allocate(512).is_none());
    }

    #[test]
    fn test_arena_free_and_coalesce() {
        let mut arena = DeviceArena::new(0u64, 1024);

        let a = arena.allocate(128).unwrap();
        let b = arena.allocate(128).unwrap();
        let c = arena.allocate(128).unwrap();

        assert_eq!(arena.live_allocations(), 3);

        arena.free(&b);
        assert_eq!(arena.live_allocations(), 2);

        arena.free(&c);
        // Coalesce: offsets 128 + 256 → continuous block {128, 384}.
        assert_eq!(arena.live_allocations(), 1);
        assert_eq!(arena.free_bytes(), 1024 - 128);

        arena.free(&a);
        assert_eq!(arena.live_allocations(), 0);
        assert_eq!(arena.free_bytes(), arena.capacity());
    }

    #[test]
    fn test_arena_reuse_freed_slot() {
        let mut arena = DeviceArena::new(0u64, 1024);

        let a = arena.allocate(128).unwrap();
        arena.free(&a);

        let b = arena.allocate(128).unwrap();
        assert_eq!(b.offset, 0); // reuses the first slot
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = DeviceArena::new(0u64, 1024);
        arena.allocate(128).unwrap();
        arena.allocate(128).unwrap();
        assert_eq!(arena.live_allocations(), 2);

        arena.reset();
        assert_eq!(arena.live_allocations(), 0);
        assert_eq!(arena.free_bytes(), arena.capacity());
    }

    #[test]
    fn test_arena_stale_cookie_rejected() {
        let mut arena = DeviceArena::new(0u64, 1024);
        let a = arena.allocate(128).unwrap();
        arena.free(&a);
        // Freeing again with the same (now stale) cookie is a no-op.
        let live_before = arena.live_allocations();
        arena.free(&a);
        assert_eq!(arena.live_allocations(), live_before);
    }

    #[test]
    fn test_arena_one_thousand_suballocations() {
        let mut arena = DeviceArena::new(0u64, 1024 * 1024); // 1 MiB

        let mut slices = Vec::new();
        for _ in 0..1000 {
            let s = arena.allocate(128).unwrap();
            slices.push(s);
        }
        assert_eq!(arena.live_allocations(), 1000);

        for s in &slices {
            arena.free(s);
        }
        assert_eq!(arena.live_allocations(), 0);
        assert_eq!(arena.free_bytes(), arena.capacity());
    }
}
