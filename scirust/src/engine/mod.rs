//! Fused SIMD and GPU matrix execution engine module.
//!
//! `hybrid` is an EXPERIMENTAL prototype. Its GPU path is quarantined
//! (feature-gated and marked experimental) because it previously loaded a
//! no-op PTX stub as production "fused GEMM" — see the mission audit
//! (docs/ELASTIC_MISSION_AUDIT.md, P0-1/P0-2). Production CUDA lives in
//! `slhav2-vram`; the hybrid path must never be the default.

pub mod hybrid;
