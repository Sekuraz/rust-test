#![cfg_attr(test, feature(test))]
#![feature(step_by)]

extern crate rand;

type NumType = f64;

const ARRAY_SIZE: usize = 10000;
const STRIDE: usize = 1;
const S: NumType = 3.5 as NumType;

type ArrayType = [NumType; ARRAY_SIZE];


fn random_array() -> ArrayType {
    let mut ret = [0 as NumType; ARRAY_SIZE];
    for i in 0..ARRAY_SIZE {
        ret[i] = rand::random::<NumType>();
        //println!("{:?}", ret[i]);
    }
    ret
}


fn main() {
    let mut res = [0 as NumType; ARRAY_SIZE];
    let a = random_array();
    let b = random_array();
    let c = random_array();

    copy(&a, &mut res);
    add(&mut res, &a, &b);
    striad(&mut res, &a, &b, S);
    vtriad(&mut res, &a, &b, &c);
}

// https://doc.rust-lang.org/std/primitive.slice.html#method.copy_from_slice
#[inline]
fn copy<T: std::marker::Copy>(src: &[T], dst: &mut [T]) {
    // This version takes much less time, it is only doing memcpy
    dst.copy_from_slice(src);

    //for i in (0..ARRAY_SIZE).step_by(STRIDE) {
    //    dst[i] = src[i];
    //}
}

#[inline]
fn add<T>(result: &mut [T], a: &[T], b: &[T])
    where T: std::marker::Copy + std::ops::Add + std::ops::Add<Output=T>
{
    assert!(result.len() == a.len() && result.len() == b.len());
    for i in (0..result.len()).step_by(STRIDE) {
        result[i] = a[i] + b[i];
    }
}

#[inline]
fn striad<T>(result: &mut [T], a: &[T], b: &[T], s: T)
    where T: std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    assert!(result.len() == a.len() && result.len() == b.len());

    for i in (0..a.len()).step_by(STRIDE) {
        result[i] = s*a[i] + b[i];
    }
}

#[inline]
fn vtriad<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
    where T: std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    assert!(result.len() == a.len() && result.len() == b.len() && result.len() == c.len());

    for i in (0..a.len()).step_by(STRIDE) {
        result[i] = c[i] * a[i] + b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate test;

    fn array_equal<T: std::cmp::PartialEq + std::fmt::Debug>(a: &[T], b: &[T]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i], b[i]);
        }
    }

    #[test]
    fn test_copy() {
        let mut dst = [0; 5];
        let src = [0, 1, 2, 3, 4];
        copy(&src, &mut dst);
        array_equal(&src, &dst)
    }

    #[test]
    fn test_add() {
        let mut result = [0; 5];
        let a = [0, 1, 2, 3, 4];
        let b = [5, 4, 3, 2, 1];
        add(&mut result, &a, &b);
        array_equal(&result, &[5; 5]);
    }

    #[test]
    fn test_striad() {
        let mut result = [0; 5];
        let a = [0, 1, 2, 3, 4];
        let b = [5, 4, 3, 2, 1];
        striad(&mut result, &a, &b, 2);
        array_equal(&result, &[5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_vtriad() {
        let mut result = [0; 5];
        let a = [0, 1, 2, 3, 4];
        let b = [5, 4, 3, 2, 1];
        let c = [0, 1, 2, 3, 4];
        vtriad(&mut result, &a, &b, &c);
        array_equal(&result, &[5, 5, 7, 11, 17]);
    }

    #[bench]
    fn bench_copy(bencher: &mut test::Bencher) {
        let mut dst = random_array();
        let src = random_array();
        bencher.iter(|| {
            copy(&src, &mut dst);
        });
    }

    #[bench]
    fn bench_add(bencher: &mut test::Bencher) {
        let mut res = [0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        bencher.iter(|| {
            add(&mut res, &a, &b);
        });
    }

    #[bench]
    fn bench_striad(bencher: &mut test::Bencher) {
        let mut res = [0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        bencher.iter(|| {
            striad(&mut res, &a, &b, S);
        });
    }

    #[bench]
    fn bench_vtriad(bencher: &mut test::Bencher) {
        let mut res = [0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        let c = random_array();
        bencher.iter(|| {
            vtriad(&mut res, &a, &b, &c);
        });
    }
}
