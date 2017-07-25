
#[macro_use]
use super::macros;

use super::tiledmatrix::*;

extern crate rayon;
use self::rayon::prelude::*;

use super::*;

#[allow(non_snake_case)]
pub fn mult(A: &TileMatrix, B: &TileMatrix, C: &mut TileMatrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.blocks_right, B.blocks_down);
    assert_eq!(A.rows, C.rows);
    assert_eq!(A.blocks_down, C.blocks_right);
    assert_eq!(B.columns, C.columns);
    assert_eq!(B.blocks_right, C.blocks_right);

    for i in 0..C.blocks_down {
        for j in 0..C.blocks_right {
            for k in 0..A.blocks_right {
                naive_simd::mult(&A[(i,k)], &B[(k, j)], &mut C[(i, j)]);
            }
        }
    }
}

generate_tests!(TileMatrix);