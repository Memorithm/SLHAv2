use crate::attention::slha_v2::SciRustSlhaTile;

/// Trait to map external cache structures to SLHA v2 tiles.
///
/// This allows users to integrate their existing KV-cache layouts into the
/// SciRust kernel by providing a way to materialise a `SciRustSlhaTile`.
pub trait ExternalCacheAdapter {
    /// Map the external structure into a `SciRustSlhaTile`.
    fn map_to_tile(&self) -> SciRustSlhaTile;
}

/// Blanket implementation for types that can be converted into a tile.
impl<T> ExternalCacheAdapter for T
where
    T: Into<SciRustSlhaTile> + Clone,
{
    fn map_to_tile(&self) -> SciRustSlhaTile {
        self.clone().into()
    }
}
