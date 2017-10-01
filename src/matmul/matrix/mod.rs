pub mod traits;
pub use self::traits::*;

pub mod standard;
pub use self::standard::Matrix;

pub mod tiled;
pub use self::tiled::TileMatrix;

pub mod transposed;
pub use self::transposed::TransposedMatrix;

pub mod simd;
pub use self::simd::SimdMatrix;