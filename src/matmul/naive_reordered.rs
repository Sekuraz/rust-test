use super::matrix::*;

use super::TOPOLOGY;

#[allow(non_snake_case)]
pub fn mult(A: &Matrix, B: &Matrix, C: &mut Matrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.rows, C.rows);
    assert_eq!(B.columns, C.columns);

    for i in 0..C.rows {
        for k in 0..A.columns {
            for j in 0..C.columns/4 {
                unsafe {
                    *C.get_unchecked_mut((i, j * 4)) += A.get_unchecked((i, k)) * B.get_unchecked((k, j * 4));
                    *C.get_unchecked_mut((i, j * 4 + 1)) += A.get_unchecked((i, k)) * B.get_unchecked((k, j * 4 + 1));
                    *C.get_unchecked_mut((i, j * 4 + 2)) += A.get_unchecked((i, k)) * B.get_unchecked((k, j * 4 + 2));
                    *C.get_unchecked_mut((i, j * 4 + 3)) += A.get_unchecked((i, k)) * B.get_unchecked((k, j * 4 + 3));
                }
            }
        }
    }
}

generate_tests!();