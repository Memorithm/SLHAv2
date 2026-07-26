use std::error::Error;

pub trait DeviceAllocation {
    fn size(&self) -> usize;
}

pub trait DeviceEngine: Sized {
    type Alloc: DeviceAllocation;
    type Error: Error;

    fn allocate(&self, size: usize) -> Result<Self::Alloc, Self::Error>;

    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &mut Self::Alloc,
        dst_offset: usize,
    ) -> Result<(), Self::Error>;

    fn copy_to_host(
        &self,
        src: &Self::Alloc,
        src_offset: usize,
        dst: &mut [u8],
    ) -> Result<(), Self::Error>;

    fn set_device(&self) -> Result<(), Self::Error>;

    fn synchronize(&self) -> Result<(), Self::Error>;
}
