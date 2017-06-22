#![cfg_attr(test, feature(test))]
#![feature(step_by)]

#![feature(associated_consts)]

extern crate core;

extern crate rand;
extern crate rayon;
extern crate simd;
extern crate num_cpus;


use core::ptr;

use rayon::prelude::*;

use simd::x86::avx::{f64x4, f32x8};


type NumType = f32;

const ARRAY_SIZE: usize = 50000;
const STRIDE: usize = 1;
const S: NumType = 3.5 as NumType;

fn random_array() -> Vec<NumType> {
    (0..ARRAY_SIZE).map(|_| rand::random::<NumType>()).collect()
}

#[allow(unused)]
fn main() {
    let mut res = vec![0 as NumType; ARRAY_SIZE];
    let a = random_array();
    let b = random_array();
    let c = random_array();

    /*
    copy(&a, &mut res);
    add(&mut res, &a, &b);
    striad(&mut res, &a, &b, S);
    vtriad(&mut res, &a, &b, &c);
    */
    /*
    vtriad_itertools(&mut res, &a, &b, &c);
    for i in 0..ARRAY_SIZE {
        println!("{}: {} * {} + {} = {}", i, a[i], c[i], b[i], res[i]);
    }
    */
    //vtriad_rayon(&mut res, &a, &b, &c);

    vtriad_rayon(&mut res, &a, &b, &c);
    //vtriad_itertools(&mut res, &a, &b, &c);
    //vtriad_simd_f64(&mut res, &a, &b, &c);
    for i in 0..ARRAY_SIZE {
        println!("{}: {} * {} + {} = {}", i, a[i], c[i], b[i], res[i]);
    }
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
fn add_itertools<T>(result: &mut Vec<T>, a: &[T], b: &[T])
    where T: std::marker::Copy + std::ops::Add<Output=T>
{
    let r = a.iter().zip(b).map(|(x, y)| *x + *y);
    result.clear();
    result.extend(r);
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

#[inline]
fn vtriad_itertools<T>(result: &mut Vec<T>, a: &[T], b: &[T], c: &[T])
    where T: std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    let r = a.iter().zip(b).zip(c).map(|((&x, &y), &z)| x * z + y);

    let result_ptr = result.as_mut_ptr();
    let result_slice = result.as_mut_slice();

    let end = r.fold(0, |index, item| { unsafe {ptr::write(result_ptr.offset(index as isize), item) }; index + 1});
    //let end = r.fold(0, |index, item| { result_slice[index] = item; index + 1});

    assert_eq!(end, result_slice.len());

    /*
    // slower but cleaner
    result.clear();
    result.extend(r);
    */
}

#[inline]
fn vtriad_rayon<T>(result: &mut Vec<T>, a: &[T], b: &[T], c: &[T])
    where T: Sync + Send + std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    /*
    let r = a.par_iter().zip(b).zip(c).map(|((&x, &y), &z)| x * z + y);
    result.clear();
    result.par_extend(r);
    */
    ///*
    let r = result.par_iter_mut().zip(a).zip(b).zip(c);

    r.for_each(|(((r, &x), &y), &z)| {
        unsafe {
            ptr::write(r as *mut T, x * z + y)
        };
    });
    //*/
}

trait SimdItem {
    type Elem: std::marker::Sized;

    #[inline]
    fn store (self, array: &mut [Self::Elem], idx: usize) where Self: core::marker::Sized;
}

impl SimdItem for f64x4 {
    type Elem = f64;

    #[inline]
    fn store (self, array: &mut [f64], idx: usize) {
        self.store(array, idx);
    }
}

impl SimdItem for f32x8 {
    type Elem = f32;

    #[inline]
    fn store (self, array: &mut [f32], idx: usize) {
        self.store(array, idx);
    }
}

trait SimdCapable where Self: core::marker::Sized {
    type SimdType: SimdItem<Elem=Self> + std::marker::Copy + std::ops::Add<Output=Self::SimdType> + std::ops::Mul<Output=Self::SimdType>;
    const CHUNK_SIZE: usize;
    type ArrayType;

    #[inline]
    fn load (array: &[Self], idx: usize) -> Self::SimdType where Self: core::marker::Sized;
}

impl SimdCapable for f64 {
    type SimdType = f64x4;
    type ArrayType = [f64; 4];
    const CHUNK_SIZE: usize = 4;

    #[inline]
    fn load (array: &[Self], idx: usize) -> Self::SimdType {
        Self::SimdType::load(array, idx)
    }
}

impl SimdCapable for f32 {
    type SimdType = f32x8;
    type ArrayType = [f32; 8];
    const CHUNK_SIZE: usize = 8;

    #[inline]
    fn load (array: &[Self], idx: usize) -> Self::SimdType {
        Self::SimdType::load(array, idx)
    }
}


#[inline]
fn vtriad_simd<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
    where T: SimdCapable + std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>,
{
    let len = result.len();
    assert_eq!(len % T::CHUNK_SIZE, 0);

    let simd = a.chunks(T::CHUNK_SIZE)
        .zip(b.chunks(T::CHUNK_SIZE))
        .zip(c.chunks(T::CHUNK_SIZE))
        .map(|((x, y), z)| (T::load(x, 0), T::load(y, 0), T::load(z, 0)));

    let r = simd.map(|(x, y, z)| x * z + y);


    let result_ptr = result.as_mut_ptr();

    let end = r.fold(0, |index, item| {
        unsafe {
            ptr::copy((&item as *const T::SimdType) as *const T, result_ptr.offset(index as isize), T::CHUNK_SIZE)
        };
        index + T::CHUNK_SIZE
    });
    /*
    let result_slice = result.as_mut_slice();

    let end = r.fold(0, |index, item| {
        item.store(result_slice, index);
        index + T::CHUNK_SIZE
    });
    */
    assert_eq!(end, len);
}

#[inline]
fn vtriad_simd_rayon<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
    where T: SimdCapable + Sync + Send + std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>,
            T::SimdType : Send
{
    let len = result.len();
    assert_eq!(len % T::CHUNK_SIZE, 0);

    let simd = result.par_chunks_mut(T::CHUNK_SIZE)
        .zip(a.par_chunks(T::CHUNK_SIZE))
        .zip(b.par_chunks(T::CHUNK_SIZE))
        .zip(c.par_chunks(T::CHUNK_SIZE))
        .map(|(((r, x), y), z)| (r, T::load(x, 0), T::load(y, 0), T::load(z, 0)));

    //let r = simd.map(|(r, x, y, z)| (r, x * z + y));

    simd.for_each(|(r, x, y, z)| {
        unsafe {
            ptr::copy((&(x * z + y) as *const T::SimdType) as *const T, r.as_mut_ptr(), T::CHUNK_SIZE)
        };
    });
}

#[allow(unused)]
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
    fn test_add_itertools() {
        let mut result = vec![0; ARRAY_SIZE];
        let a = [0, 1, 2, 3, 4];
        let b = [5, 4, 3, 2, 1];
        add_itertools(&mut result, &a, &b);
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

    #[test]
    fn test_vtriad_itertools() {
        let mut result = vec![0; 5];
        let a = [0, 1, 2, 3, 4];
        let b = [5, 4, 3, 2, 1];
        let c = [0, 1, 2, 3, 4];
        vtriad_itertools(&mut result, &a, &b, &c);
        array_equal(&result, &[5, 5, 7, 11, 17]);

        let mut result_iter = vec![0 as NumType; ARRAY_SIZE];
        let mut result_std = vec![0 as NumType; ARRAY_SIZE];

        let x = random_array();
        let y = random_array();
        let z = random_array();
        vtriad_itertools(&mut result_iter, &x, &y, &z);
        vtriad(&mut result_std, &x, &y, &z);
        array_equal(&result_std, &result_iter);
    }

    #[test]
    fn test_vtriad_rayon() {
        let mut result = vec![0; 5];
        let a = [0, 1, 2, 3, 4];
        let b = [5, 4, 3, 2, 1];
        let c = [0, 1, 2, 3, 4];
        vtriad_rayon(&mut result, &a, &b, &c);
        array_equal(&result, &[5, 5, 7, 11, 17]);
    }

    #[test]
    fn test_vtriad_simd() {
        let mut result = vec![0.; 8];
        let a = [0., 1., 2., 3., 4., 5., 6., 7.];
        let b = [5., 4., 3., 2., 1., 0., 1., 2.];
        let c = [0., 1., 2., 3., 4., 5., 6., 7.];
        vtriad_simd(&mut result, &a, &b, &c);
        array_equal(&result, &[5., 5., 7., 11., 17., 25., 37., 51.]);
    }

    #[test]
    fn test_vtriad_simd_rayon() {
        let mut result = vec![0.; 8];
        let a = [0., 1., 2., 3., 4., 5., 6., 7.];
        let b = [5., 4., 3., 2., 1., 0., 1., 2.];
        let c = [0., 1., 2., 3., 4., 5., 6., 7.];
        vtriad_simd_rayon(&mut result, &a, &b, &c);
        array_equal(&result, &[5., 5., 7., 11., 17., 25., 37., 51.]);
    }

    //#[bench]
    fn bench_copy(bencher: &mut test::Bencher) {
        let mut dst = random_array();
        let src = random_array();
        bencher.iter(|| {
            test::black_box(copy(&src, &mut dst));
        });
    }

    #[bench]
    fn bench_add(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        bencher.iter(|| {
            test::black_box(add(&mut res, &a, &b));
        });
    }

    #[bench]
    fn bench_add_itertools(bencher: &mut test::Bencher) {
        let mut result = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        bencher.iter(|| {
            test::black_box(add_itertools(&mut result, &a, &b));
        });
    }

    #[bench]
    fn bench_striad(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        bencher.iter(|| {
            test::black_box(striad(&mut res, &a, &b, S));
        });
    }

    #[bench]
    fn bench_vtriad(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        let c = random_array();
        bencher.iter(|| {
            test::black_box(vtriad(&mut res, &a, &b, &c));
        });
    }

    #[bench]
    fn bench_vtriad_itertools(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        let c = random_array();
        vtriad_itertools(&mut res, &a, &b, &c);
        res[ARRAY_SIZE - 1] += 1.0; // If res of the method is not used there is no code emitted
        bencher.iter(|| {
            test::black_box(vtriad_itertools(&mut res, &a, &b, &c));
        });
    }

    #[bench]
    fn bench_vtriad_rayon(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        let c = random_array();
        bencher.iter(|| {
            test::black_box(vtriad_rayon(&mut res, &a, &b, &c));
        });
    }

    //#[bench]
    fn bench_vtriad_simd(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        let c = random_array();
        bencher.iter(|| {
            test::black_box(vtriad_simd(&mut res, &a, &b, &c));
        });
    }

    #[bench]
    fn bench_vtriad_simd_rayon(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = random_array();
        let b = random_array();
        let c = random_array();
        bencher.iter(|| {
            test::black_box(vtriad_simd_rayon(&mut res, &a, &b, &c));
        });
    }
}
