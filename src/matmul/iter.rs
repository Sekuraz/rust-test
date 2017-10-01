
#[macro_use]
use super::macros;

use super::*;

#[allow(non_snake_case)]
pub fn mult(A: &Matrix, B: &TransposedMatrix, C: &mut Matrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.rows, C.rows);
    assert_eq!(B.columns, C.columns);

    for i in 0..C.columns {
        let a_columns = A.columns;
        let c_rows = C.rows;

        A.data
            .chunks(a_columns)
            .zip(B.data.chunks(a_columns).cycle().skip(i))
            .map(|(row, column)| {
                row.iter().zip(column).map(|(a, b)| a * b).fold(
                    0.0,
                    |acc, item| {
                        acc + item
                    },
                )
            })
            .enumerate()
            .map(|(ind, result)| ((ind, (ind + i) % c_rows), result))
            .for_each(|(index, result)| unsafe { *C.get_unchecked_mut(index) = result });
    }
}

generate_tests!(Matrix, TransposedMatrix, Matrix);
