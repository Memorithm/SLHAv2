// Every unsafe block must state its precondition. Permanent gate: clippy
// runs with -D warnings in CI, so an undocumented block fails the build.
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod backends;
pub mod codec;
pub mod elastic_cache;
pub mod mem;
pub mod pipeline;
pub mod traits;
