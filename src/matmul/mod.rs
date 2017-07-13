#[macro_use]
mod macros;

use super::TOPOLOGY;

pub mod naive;
pub mod naive_unchecked;
pub mod naive_reordered;
pub mod naive_simd;
pub mod naive_rayon;

pub mod matrix;
pub use self::matrix::Matrix;

#[cfg(test)]
mod test {

    use super::*;
    use std::fs::File;
    use std::io::prelude::*;

    #[test]
    fn test_equal () {
        let n = 256;

        let a = Matrix::random(n, n);
        let b = Matrix::random(n, n);
        let mut compare = Matrix::zero(n, n);
        naive::mult(&a, &b, &mut compare);

        let mut res = Matrix::zero(n, n);
        naive_unchecked::mult(&a, &b, &mut res);
        assert_eq!(compare, res);

        naive_reordered::mult(&a, &b, &mut res);
        assert_eq!(compare, res);

        naive_simd::mult(&a, &b, &mut res);
        File::create("/tmp/a").unwrap().write_all(format!("{:?}", a).as_bytes());
        File::create("/tmp/b").unwrap().write_all(format!("{:?}", b).as_bytes());

        File::create("/tmp/simd").unwrap().write_all(format!("{:?}", res).as_bytes());
        File::create("/tmp/comp").unwrap().write_all(format!("{:?}", compare).as_bytes());

        assert_eq!(compare, res);

        naive_rayon::mult(&a, &b, &mut res);
        assert_eq!(compare, res);
    }
}


