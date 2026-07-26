use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::traits::{DeviceAllocation, DeviceEngine};

pub struct ArenaSlice {
    pub offset: u64,
    pub size: usize,
    pub cookie: u64,
}

impl ArenaSlice {
    pub fn is_valid(&self) -> bool {
        self.cookie != 0
    }
}

pub trait RawDeviceAllocator {
    type Alloc: DeviceAllocation;
    type Error: std::error::Error;

    fn allocate(&self, size: usize) -> Result<Self::Alloc, Self::Error>;
}

pub struct DeviceArena<A: DeviceAllocation> {
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

#[allow(dead_code)]
struct AllocBlock {
    offset: u64,
    size: u64,
    cookie: u64,
}

impl<A: DeviceAllocation> DeviceArena<A> {
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
        self.alloc_map
            .values()
            .map(|b| b.size)
            .sum()
    }

    pub fn allocate(&mut self, size: u64) -> Option<ArenaSlice> {
        let aligned = size.next_power_of_two().max(128);

        let idx = self
            .free_list
            .iter()
            .position(|fb| fb.size >= aligned)?;

        let block = self.free_list.remove(idx);
        let cookie = self.next_cookie.fetch_add(1, Ordering::Relaxed);

        let slice = ArenaSlice {
            offset: block.offset,
            size: aligned as usize,
            cookie,
        };

        self.alloc_map.insert(
            cookie,
            AllocBlock {
                offset: block.offset,
                size: aligned,
                cookie,
            },
        );

        let remainder = block.size.saturating_sub(aligned);
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
                let merged_size = a.size.max(b.offset + b.size - a.offset);
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

impl<A: DeviceAllocation> DeviceArena<A> {
    pub fn copy_tiles_to_device<E: DeviceEngine<Alloc = A>>(
        &mut self,
        engine: &E,
        tiles: &[crate::mem::tile::SerializedTile],
        dst: &mut ArenaSlice,
    ) -> Result<(), E::Error> {
        let total = tiles.len() * crate::codec::TILE_BYTES;
        assert!(
            total as u64 <= dst.size as u64,
            "tile data ({total}) exceeds slice capacity ({})",
            dst.size
        );

        let mut buf = vec![0u8; total];
        for (i, tile) in tiles.iter().enumerate() {
            let off = i * crate::codec::TILE_BYTES;
            buf[off..off + crate::codec::TILE_BYTES].copy_from_slice(&tile.0);
        }

        engine.copy_to_device(&buf, &mut self.backing, dst.offset as usize)
    }
}
