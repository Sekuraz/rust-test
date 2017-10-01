
#[macro_use]
use super::macros;

use super::*;

#[allow(non_snake_case)]
pub fn mult(A: &Matrix, B: &TransposedMatrix, C: &mut Matrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.rows, C.rows);
    assert_eq!(B.columns, C.columns);

    assert_eq!(A.columns % 2, 0);
    assert_eq!(A.rows % 2, 0);
    assert_eq!(B.columns % 2, 0);

    for i in (0..C.rows).step_by(2) {
        for j in (0..C.columns).step_by(2) {
            let mut result1 = C[(i, j)];
            let mut result2 = C[(i + 1, j)];
            let mut result3 = C[(i, j + 1)];
            let mut result4 = C[(i + 1, j + 1)];

            for k in 0..A.columns {
                result1 += A[(i, k)] * B[(k, j)];
                result2 += A[(i + 1, k)] * B[(k, j)];
                result3 += A[(i, k)] * B[(k, j + 1)];
                result4 += A[(i + 1, k)] * B[(k, j + 1)];
            }
            C[(i, j)] = result1;
            C[(i + 1, j)] = result2;
            C[(i, j + 1)] = result3;
            C[(i + 1, j + 1)] = result4;
        }
    }
}

generate_tests!(Matrix, TransposedMatrix, Matrix);