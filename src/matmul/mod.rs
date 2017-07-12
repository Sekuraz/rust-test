#[macro_use]
mod macros;

use super::TOPOLOGY;

pub mod naive;
pub mod naive_unchecked;
pub mod naive_reordered;
pub mod naive_simd;

pub mod matrix;
pub use self::matrix::Matrix;


