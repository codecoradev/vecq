//! vecq — training-free 4-bit vector quantization and search.
//!
//! The "SQLite profile" for vector storage: single file, embedded,
//! deterministic, no training pass. Vectors are rotated with a randomized
//! Hadamard transform (seeded, stored in the file header) so coordinates
//! become approximately N(0,1), then quantized with precomputed Lloyd-Max
//! 4-bit tables and nibble-packed two dimensions per byte.

pub mod format;
pub mod lloyd;
pub mod rhdh;
pub mod store;
pub mod view;

pub use store::{cosine_f32, PreparedQuery, VecqIndex};
pub use view::VecqView;
