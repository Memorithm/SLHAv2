use std::mem::size_of;

pub unsafe fn bytes_as<T: Copy>(src: &[u8]) -> &T {
    assert!(src.len() >= size_of::<T>());
    &*(src.as_ptr() as *const T)
}

pub fn bytes_as_slice<T: Copy>(src: &[u8]) -> &[T] {
    assert_eq!(src.len() % size_of::<T>(), 0);
    let len = src.len() / size_of::<T>();
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const T, len) }
}

pub fn bytes_as_slice_mut<T: Copy>(src: &mut [u8]) -> &mut [T] {
    assert_eq!(src.len() % size_of::<T>(), 0);
    let len = src.len() / size_of::<T>();
    unsafe { std::slice::from_raw_parts_mut(src.as_mut_ptr() as *mut T, len) }
}

pub fn copy_into_slice(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    dst.copy_from_slice(src);
}
