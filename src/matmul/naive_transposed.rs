
#[macro_use]
use super::macros;

use super::matrix::*;


#[allow(non_snake_case)]
pub fn mult(A: &Matrix, B: &TransposedMatrix, C: &mut Matrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.rows, C.rows);
    assert_eq!(B.columns, C.columns);

    for i in 0..C.rows {
        for j in 0..C.columns {
            let mut tmp = 0.0;
            for k in 0..A.columns {
                unsafe {
                    tmp += A.get_unchecked((i, k)) * B.get_unchecked((k, j));
                }
            }
            C[(i, j)] = tmp;
        }
    }
}

generate_tests!(Matrix, TransposedMatrix, Matrix);