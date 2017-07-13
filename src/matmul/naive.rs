
#[macro_use]
use super::macros;

use super::matrix::*;

use super::TOPOLOGY;

#[allow(non_snake_case)]
pub fn mult(A: &Matrix, B: &Matrix, C: &mut Matrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.rows, C.rows);
    assert_eq!(B.columns, C.columns);

    C.reset();

    for i in 0..C.rows {
        for j in 0..C.columns {
            for k in 0..A.columns {
                C[(i,j)] += A[(i, k)] * B[(k, j)];
            }
        }
    }
}

generate_tests!();