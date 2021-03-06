
macro_rules! new_matrix {
    (Matrix, $mat:expr) => {{
        $mat
    }};
    (TileMatrix, $mat:expr) => {{
        TileMatrix::from($mat)
    }};
    (TransposedMatrix, $mat:expr) => {{
        TransposedMatrix::from($mat)
    }}
}

#[macro_export]
macro_rules! generate_tests {
    () => {
        generate_tests!(Matrix);
    };
    ($mat_type:ident) => {
        generate_tests!($mat_type, $mat_type, $mat_type);
    };
    ($mat_type_A:ident, $mat_type_B:ident, $mat_type_C:ident) => {
        #[cfg(test)]
        mod tests {

            extern crate simd;
            use self::simd::x86::avx::f64x4;


            use super::mult;
            use super::super::{ Matrix, TileMatrix, TransposedMatrix };

            extern crate test;

            #[test]
            fn test_mult_4x4_4x4() {
                let a = new_matrix!($mat_type_A, Matrix::new_aligned(4, 4, vec![f64x4::splat(1.); 4]));
                let b = new_matrix!($mat_type_B, Matrix::new_aligned(4, 4, vec![f64x4::splat(1.); 4]));
                let mut c = new_matrix!($mat_type_C, Matrix::zero(4, 4));

                mult(&a, &b, &mut c);
                assert_eq!(c, new_matrix!($mat_type_C, Matrix::new_aligned(4, 4, vec![f64x4::splat(4.); 4])));
            }
/*
            #[test]
            fn test_mult_8x4_4x8() {
                let a = new_matrix!($mat_type_A, Matrix::new(8, 4, (1...8*4).map(|i| i as f64).collect()));
                let b = new_matrix!($mat_type_B, Matrix::new(4, 8, (1...8*4).map(|i| i as f64).collect()));
                let mut c = new_matrix!($mat_type_C, Matrix::new(8, 8, vec![0.; 8*8]));

                mult(&a, &b, &mut c);
                assert_eq!(c, new_matrix!($mat_type_C, Matrix::new(8, 8, vec![
                170., 180., 190., 200., 210., 220., 230., 240.,
                378., 404., 430., 456., 482., 508., 534., 560.,
                586., 628., 670., 712., 754., 796., 838., 880.,
                794., 852., 910., 968., 1026., 1084., 1142., 1200.,
                1002., 1076., 1150., 1224., 1298., 1372., 1446., 1520.,
                1210., 1300., 1390., 1480., 1570., 1660., 1750., 1840.,
                1418., 1524., 1630., 1736., 1842., 1948., 2054., 2160.,
                1626., 1748., 1870., 1992., 2114., 2236., 2358., 2480.])));
            }*/

            #[bench]
            fn bench_256x256_256x256(bencher: &mut test::Bencher) {
                let n = 512;
                let a = new_matrix!($mat_type_A, Matrix::random(n, n));
                let b = new_matrix!($mat_type_B, Matrix::random(n, n));
                let mut c = new_matrix!($mat_type_C, Matrix::zero(n, n));

                bencher.iter(|| {
                    mult(&a, &b, &mut c);
                });

                //println!();
                //println!("FLOP: {}", 2*n*n*n);
                //println!("ns/iter: {}", bencher.summary.unwrap().mean);
                //println!("FLOP/s: {}", 2*n*n*n as f64 / bencher.summary.unwrap().mean * 1_000_000_000 as f64)
            }
        }
    }
}
