
#[macro_use]
use super::macros;

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

    (0..C.blocks_down).into_par_iter().for_each(|i| {
        let c_data = C.data.as_ptr() as *mut Matrix;
        for j in 0..C.blocks_right {
            for k in 0..A.blocks_right {
                naive_simd::mult(&A[(i,k)], &B[(k, j)], unsafe { &mut *c_data.offset((i * C.blocks_right + j) as isize)});
            }
        }
    });
}

generate_tests!(TileMatrix);