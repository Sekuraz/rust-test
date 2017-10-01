
#[macro_use]
use super::macros;

use std;
use self::std::ptr;

extern crate simd;
use self::simd::x86::avx::f64x4;

extern crate rayon;
use self::rayon::prelude::*;

use super::*;

#[allow(non_snake_case)]
pub fn mult(A: &Matrix, B: &TransposedMatrix, C: &mut Matrix) {
    assert_eq!(A.columns, B.rows);
    assert_eq!(A.rows, C.rows);
    assert_eq!(B.columns, C.columns);

    assert_eq!(C.rows % 4, 0);
    assert_eq!(A.columns % 4, 0);
    assert_eq!(C.columns % 4, 0);

    assert!(A.is_aligned() && B.is_aligned() && C.is_aligned());

    (0..C.rows).into_par_iter().for_each(|i| {
        let c_data = C.data.as_ptr() as *mut f64;
        for j in 0..C.columns {
            let mut result1 = 0.0;

            let mut indexA = i * C.columns;
            let mut indexB = j * C.rows; // TODO

            let mut res = f64x4::splat(0.0);

            unsafe {
                /*asm!(
                    "
                    movl $0, %%eax
                    vmovapd %0, %%ymm4
                    1:
                    vmovapd (%3, %1, 8), %%ymm1
                    vmovapd (%4, %2, 8), %%ymm2
                    vmulpd %%ymm1, %%ymm2, %%ymm3
                    vaddpd %%ymm3, %%ymm4, %%ymm4
                    addq $4, %1
                    addq $4, %2
                    addl $4, %%eax
                    cmp %5, %%eax
                    jl 1b
                    vmovapd %%ymm4, %0
                    "
                    :"+m"(temp1), "+r"(indexA), "+r"(indexB)
                    :"r"(A.data.as_ptr()), "r"(B.data.as_ptr()), "r"(C.rows)
                    :"%ymm1", "%ymm2", "%ymm3", "%ymm4", "%eax"
                    :
                );*/
                if C.columns % 16 == 0 {
                    asm!(
                        "
                        movq $$0, %rdx
                        1:
                        vmovapd ($1, $3, 8), %ymm1
                        vmovapd ($2, $4, 8), %ymm2
                        vfmadd231pd %ymm1, %ymm2, $0

                        addq $$4, $3
                        addq $$4, $4
                        addq $$4, %rdx
                        cmp $5, %rdx
                        jl  1b
                        "
                        : "+x"(res)
                        : "r" (A.data.as_ptr()),
                          "r" (B.data.as_ptr() as *mut f64),
                          "r" (indexA),
                          "r" (indexB),
                          "r" (A.columns)
                        : "rdx", "ymm1", "ymm2", "ymm3"
                    );
                }
                else {
                    asm!(
                        "
                        movq $$0, %rdx
                        1:
                        vmovapd ($1, $3, 8), %ymm1
                        vmovapd ($2, $4, 8), %ymm2
                        vmulpd %ymm1, %ymm2, %ymm3
                        vaddpd %ymm3, $0, $0

                        addq $$4, $3
                        addq $$4, $4
                        addq $$4, %rdx
                        cmp $5, %rdx
                        jl  1b
                        "
                        : "+x"(res)
                        : "r" (A.data.as_ptr()),
                          "r" (B.data.as_ptr() as *mut f64),
                          "r" (indexA),
                          "r" (indexB),
                          "r" (A.columns)
                        : "rdx", "ymm1", "ymm2", "ymm3"
                    );
                }
            }
            //
            //            asm! (
            //                 "movl $0, %%eax\n\t"
            //                 //move temporary result vector to registers
            //                 "vmovapd %0, %%ymm4;\n\t"
            //
            //                 "1:\n\t"
            //
            //                 //load components from A
            //                 "vmovapd (%3, %1, 8), %%ymm1;\n\t"
            //                 //load components from B
            //                 "vmovapd (%4, %2, 8), %%ymm2;\n\t"
            //
            //                 //multiply componentwise
            //                 "vmulpd %%ymm1, %%ymm2, %%ymm3;\n\t"
            //                 //add to temporary result vector
            //                 "vaddpd %%ymm3, %%ymm4, %%ymm4;\n\t"
            //
            //                 //calculate index + k
            //                 "addq $4, %1;\n\t"
            //                 "addq $4, %2;\n\t"
            //
            //                 //is k < n? loop: exit
            //                 "addl $4, %%eax;\n\t"
            //                 "cmp %5, %%eax;\n\t" // calculates k - n and stores sign as flag
            //                 "jl 1b;\n\t"
            //
            //                 //move temporary result vector back to memory
            //                 "vmovapd %%ymm4, %0;\n\t"
            //
            //                 //output: 0, 1, 2
            //                 :"+m"(temp1), "+r"(indexA), "+r"(indexB)
            //                 //input: 3, 4, 5
            //                 :"r"(A), "r"(B), "r"(n)
            //                 :"%ymm1", "%ymm2", "%ymm3", "%ymm4", "%eax"
            //                 :
            //            );

            for k in 0..4 {
                result1 += res.extract(k);
            }
            unsafe {
                ptr::write(c_data.offset((i * C.columns + j) as isize), result1);}
        }
    });
}

generate_tests!(Matrix, TransposedMatrix, Matrix);