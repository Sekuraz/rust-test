#[macro_use]
mod macros;

use super::TOPOLOGY;

pub mod naive;
pub mod naive_unchecked;
pub mod naive_reordered;
pub mod naive_transposed;
pub mod naive_simd;
pub mod naive_rayon;

pub mod iter;

pub mod blocked;

pub mod tiled;
pub mod tiled_rayon;

pub mod asm;

pub mod matrix;
pub use self::matrix::*;

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

        let a_t = TileMatrix::from(a.clone());
        let b_t = TileMatrix::from(b.clone());
        let compare_t = TileMatrix::from(compare.clone());

        let mut res = Matrix::zero(n, n);
        let mut res_t = TileMatrix::from(res.clone());

        #[allow(non_snake_case)]
        let bT = TransposedMatrix::from(b.clone());


        naive_unchecked::mult(&a, &b, &mut res);
        assert_eq!(compare, res);
        res.reset();

        naive_reordered::mult(&a, &b, &mut res);
        assert_eq!(compare, res);
        res.reset();

        naive_simd::mult(&a, &b, &mut res);
        assert_eq!(compare, res);
        res.reset();

        naive_rayon::mult(&a, &b, &mut res);
        assert_eq!(compare, res);
        res.reset();

        blocked::mult(&a, &bT, &mut res);
        assert_eq!(compare, res);
        res.reset();

        iter::mult(&a, &bT, &mut res);
        assert_eq!(compare, res);
        res.reset();

        tiled::mult(&a_t, &b_t, &mut res_t);
        assert_eq!(compare_t, res_t);
        res_t.reset();

        tiled_rayon::mult(&a_t, &b_t, &mut res_t);
        assert_eq!(compare_t, res_t);
        res_t.reset();

        asm::mult(&a, &bT, &mut res);
        assert_eq!(compare, res);
        res.reset();

//        File::create("/tmp/a").unwrap().write_all(format!("{}", a_t).as_bytes());
//        File::create("/tmp/b").unwrap().write_all(format!("{}", b_t).as_bytes());
//
//        File::create("/tmp/simd").unwrap().write_all(format!("{}", res_t).as_bytes());
//        File::create("/tmp/comp").unwrap().write_all(format!("{}", compare_t).as_bytes());
//        File::create("/tmp/comp_orig").unwrap().write_all(format!("{}", compare).as_bytes());


    }
}


