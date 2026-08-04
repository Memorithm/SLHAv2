//! Per-codec quantisers, one module per codec.

pub mod int4;
pub mod mix3;
pub mod mixed;
pub mod nf4;
pub mod tq3;

pub use int4::{quantize_latent, quantize_latent_grouped};
pub use mix3::quantize_latent_mix3;
pub use mixed::quantize_latent_mixed;
pub use nf4::quantize_latent_nf4;
pub use tq3::quantize_latent_tq3;
