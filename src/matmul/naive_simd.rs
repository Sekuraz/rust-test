
extern crate simd;
use self::simd::x86::avx::f64x4;

extern crate core;
use std;
use self::core::ptr;

use super::matrix::*;

use super::TOPOLOGY;


#[allow(non_snake_case)]
pub fn mult(A: &Matrix, B: &Matrix, C: &mut Matrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.rows, C.rows);
    assert_eq!(B.columns, C.columns);

    assert_eq!(C.rows % 4, 0);
    assert_eq!(A.columns % 4, 0);
    assert_eq!(C.columns % 4, 0);


    C.reset();

    let b_ptr = B.data.as_ptr() as *const f64x4;
    let c_ptr = C.data.as_ptr() as *mut f64x4;

    for i in 0..C.rows {
        for k in 0..A.columns {
            if C.columns % 16 == 0 && A.is_aligned() && B.is_aligned() && C.is_aligned() {
                for j in 0..C.columns/16 {
                    unsafe {
                        let a = f64x4::splat(*A.get_unchecked((i, k)));

                        let b_ind = (k * B.columns / 4 + j * 4) as isize;

                        let b_1 = *b_ptr.offset(b_ind);
                        let b_2 = *b_ptr.offset(b_ind + 1);
                        let b_3 = *b_ptr.offset(b_ind + 2);
                        let b_4 = *b_ptr.offset(b_ind + 3);

                        let c_ind = (i * B.columns / 4 + j * 4) as isize;

                        let c_1 = c_ptr.offset(c_ind);
                        let c_2 = c_ptr.offset(c_ind + 1);
                        let c_3 = c_ptr.offset(c_ind + 2);
                        let c_4 = c_ptr.offset(c_ind + 3);

                        ptr::write(c_1, a * b_1 + *c_1);
                        ptr::write(c_2, a * b_2 + *c_2);
                        ptr::write(c_3, a * b_3 + *c_3);
                        ptr::write(c_4, a * b_4 + *c_4);
                    }
                }
            }
            else {
                for j in 0..C.columns/4 {
                    unsafe {
                        let a = f64x4::splat(*A.get_unchecked((i, k)));
                        let b = f64x4::load(&B.data, k * B.columns + j * 4);
                        let c = f64x4::load(&C.data, i * C.columns + j * 4);

                        let result = a * b + c;
                        ptr::write(c_ptr.offset((i * C.columns / 4 + j) as isize), result);
                    }
                }
            }

        }
    }
}

generate_tests!();