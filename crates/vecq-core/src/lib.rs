//! vecq — training-free vector quantization and search at configurable
//! width (4/5/6-bit, default 5-bit).
//!
//! The "SQLite profile" for vector storage: single file, embedded,
//! deterministic, no training pass. Vectors are rotated with a randomized
//! Hadamard transform (seeded, stored in the file header) so coordinates
//! become approximately N(0,1), then quantized with precomputed Lloyd-Max
//! tables and bit-packed LSB-first. [`view::VecqView`] serves the same
//! file format zero-copy from any byte owner (e.g. an mmap).

pub mod format;
pub mod lloyd;
pub mod rhdh;
pub mod store;
pub mod view;

pub use store::{cosine_f32, PreparedQuery, VecqIndex};
pub use view::VecqView;
