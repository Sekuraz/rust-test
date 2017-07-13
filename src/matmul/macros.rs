#[macro_export]
macro_rules! generate_tests {
    () => (
        #[cfg(test)]
        mod tests {
            use super::*;

            extern crate test;

            #[test]
            fn test_mult_4x4_4x4() {
                let a = Matrix::new(4, 4, vec![1.; 16]);
                let b = Matrix::new(4, 4, vec![1.; 16]);
                let mut c = Matrix::new(4, 4, vec![0.; 16]);

                mult(&a, &b, &mut c);
                assert_eq!(c, Matrix::new(4, 4, vec![4.; 16]));
            }

            #[test]
            fn test_mult_8x4_4x8() {
                let a = Matrix::new(8, 4, (1...8*4).map(|i| i as f64).collect());
                let b = Matrix::new(4, 8, (1...8*4).map(|i| i as f64).collect());
                let mut c = Matrix::new(8, 8, vec![0.; 8*8]);

                mult(&a, &b, &mut c);
                assert_eq!(c, Matrix::new(8, 8, vec![
                170., 180., 190., 200., 210., 220., 230., 240.,
                378., 404., 430., 456., 482., 508., 534., 560.,
                586., 628., 670., 712., 754., 796., 838., 880.,
                794., 852., 910., 968., 1026., 1084., 1142., 1200.,
                1002., 1076., 1150., 1224., 1298., 1372., 1446., 1520.,
                1210., 1300., 1390., 1480., 1570., 1660., 1750., 1840.,
                1418., 1524., 1630., 1736., 1842., 1948., 2054., 2160.,
                1626., 1748., 1870., 1992., 2114., 2236., 2358., 2480.]));
            }

            #[bench]
            fn bench_256x256_256x256(bencher: &mut test::Bencher) {
                let n = 256;
                let a = Matrix::random(n, n);
                let b = Matrix::random(n, n);
                let mut c = Matrix::zero(n, n);

                bencher.iter(|| {
                    mult(&a, &b, &mut c);
                });

                //println!();
                //println!("FLOP: {}", 2*n*n*n);
                //println!("ns/iter: {}", bencher.summary.unwrap().mean);
                //println!("FLOP/s: {}", 2*n*n*n as f64 / bencher.summary.unwrap().mean * 1_000_000_000 as f64)
            }
        }
    )
}