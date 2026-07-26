use std::collections::HashMap;
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

pub struct DeviceArena<A> {
    backing: A,
    capacity: u64,
    free_list: Vec<FreeBlock>,
    alloc_map: HashMap<u64, AllocBlock>,
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
        Self {
            backing,
            capacity,
            free_list: vec![FreeBlock {
                offset: 0,
                size: capacity,
            }],
            alloc_map: HashMap::new(),
            next_cookie: AtomicU64::new(1),
        }
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
        self.alloc_map.values().map(|b| b.size).sum()
    }

    pub fn free_bytes(&self) -> u64 {
        self.free_list.iter().map(|fb| fb.size).sum()
    }

    pub fn live_allocations(&self) -> usize {
        self.alloc_map.len()
    }

    pub fn allocate(&mut self, size: u64) -> Option<ArenaSlice> {
        if size == 0 {
            return None;
        }
        // Round up to 256-byte alignment with checked arithmetic.
        let aligned = size.checked_add(255)? & !255;

        let idx = self.free_list.iter().position(|fb| fb.size >= aligned)?;

        let block = self.free_list.remove(idx);
        let cookie = self.next_cookie.fetch_add(1, Ordering::Relaxed);

        let slice = ArenaSlice {
            offset: block.offset,
            size: aligned,
            cookie,
        };

        self.alloc_map.insert(
            cookie,
            AllocBlock {
                offset: block.offset,
                size: aligned,
            },
        );

        let remainder = block.size.checked_sub(aligned).unwrap_or(0);
        if remainder > 0 {
            self.free_list.push(FreeBlock {
                offset: block.offset + aligned,
                size: remainder,
            });
        }

        Some(slice)
    }

    pub fn free(&mut self, slice: &ArenaSlice) {
        if slice.cookie == 0 {
            return;
        }
        if let Some(block) = self.alloc_map.remove(&slice.cookie) {
            self.free_list.push(FreeBlock {
                offset: block.offset,
                size: block.size,
            });
            self.coalesce();
        }
    }

    fn coalesce(&mut self) {
        self.free_list.sort_by_key(|fb| fb.offset);
        let mut i = 0;
        while i + 1 < self.free_list.len() {
            let a = self.free_list[i];
            let b = self.free_list[i + 1];
            if a.offset + a.size >= b.offset {
                let merged_end = a.offset + a.size;
                let b_end = b.offset + b.size;
                let merged_size = if merged_end >= b_end {
                    a.size
                } else {
                    b_end - a.offset
                };
                self.free_list[i] = FreeBlock {
                    offset: a.offset,
                    size: merged_size,
                };
                self.free_list.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    pub fn reset(&mut self) {
        self.free_list.clear();
        self.free_list.push(FreeBlock {
            offset: 0,
            size: self.capacity,
        });
        self.alloc_map.clear();
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
        assert_eq!(a.size, 256); // aligned to 256

        let b = arena.allocate(128).unwrap();
        assert_eq!(b.offset, 256);
        assert_eq!(b.size, 256);

        assert_eq!(arena.live_allocations(), 2);
        assert_eq!(arena.used_bytes(), 512);
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
        // Coalesce: offsets 256 + 512 → continuous block {256, 768}
        assert_eq!(arena.live_allocations(), 1);
        assert_eq!(arena.free_bytes(), 768);

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
