
extern crate simd;

use self::simd::x86::avx::f64x4;

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
                    let a = f64x4::splat(*A.get_unchecked((i, k)));
                    let b = f64x4::load(&B.data, k * B.columns + j * 4);
                    let c = f64x4::load(&C.data, i * C.columns + j * 4);

                    let result = a * b + c;
                    result.store(&mut C.data, i * C.columns + j * 4);
                }
            }
        }
    }
}

generate_tests!();