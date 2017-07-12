extern crate core;

extern crate rayon;
extern crate simd;
extern crate hwloc;

use std;
use self::core::ptr;

use self::rayon::prelude::*;

use self::simd::x86::avx::{f64x4, f32x8};

use self::hwloc::{Topology, ObjectType};

use TOPOLOGY;
use random_array;


pub type NumType = f64;

pub const ARRAY_SIZE: usize = 4000000;
pub const STRIDE: usize = 1;

pub const S: NumType = 3.5 as NumType;


pub fn prepare_arrays() -> (Vec<NumType>, Vec<NumType>, Vec<NumType>, Vec<NumType>){
    let array_length = match std::env::var("ARRAY_SIZE") {
        Ok(len) => len.parse::<usize>().expect("ARRAY_SIZE env variable must be the array length"),
        Err(_)  => 4000000
    };

    return (vec![0 as NumType; array_length], random_array(), random_array(), random_array())
}

// https://doc.rust-lang.org/std/primitive.slice.html#method.copy_from_slice
pub fn copy<T: std::marker::Copy>(src: &[T], dst: &mut [T]) {
    // This version takes much less time, it is only doing memcpy
    dst.copy_from_slice(src);

    //for i in (0..ARRAY_SIZE).step_by(STRIDE) {
    //    dst[i] = src[i];
    //}
}

pub fn add<T>(result: &mut [T], a: &[T], b: &[T])
    where T: std::marker::Copy + std::ops::Add + std::ops::Add<Output=T>
{
    assert!(result.len() == a.len() && result.len() == b.len());
    for i in (0..result.len()).step_by(STRIDE) {
        result[i] = a[i] + b[i];
    }
}

pub fn add_itertools<T>(result: &mut Vec<T>, a: &[T], b: &[T])
    where T: std::marker::Copy + std::ops::Add<Output=T>
{
    let r = a.iter().zip(b).map(|(x, y)| *x + *y);
    result.clear();
    result.extend(r);
}

pub fn striad<T>(result: &mut [T], a: &[T], b: &[T], s: T)
    where T: std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    assert!(result.len() == a.len() && result.len() == b.len());

    for i in (0..a.len()).step_by(STRIDE) {
        result[i] = s*a[i] + b[i];
    }
}

pub fn vtriad<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
    where T: std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    assert!(result.len() == a.len() && result.len() == b.len() && result.len() == c.len());

    for i in (0..a.len()).step_by(STRIDE) {
        result[i] = c[i] * a[i] + b[i];
    }
}


pub fn vtriad_itertools<T>(result: &mut Vec<T>, a: &[T], b: &[T], c: &[T])
    where T: std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    let r = a.iter().zip(b).zip(c).map(|((&x, &y), &z)| x * z + y);

    let result_ptr = result.as_mut_ptr();
    let result_slice = result.as_mut_slice();

    let end = r.fold(0, |index, item| { unsafe {core::ptr::write(result_ptr.offset(index as isize), item) }; index + 1});
    assert_eq!(end, result_slice.len());
}

pub fn vtriad_itertools_2<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
    where T: std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    let r = result.iter_mut().zip(a).zip(b).zip(c);

    for (((r, &x), &y), &z) in r {
        *r = x * z + y;
    }
}

pub fn vtriad_rayon<T>(result: &mut Vec<T>, a: &[T], b: &[T], c: &[T])
    where T: Sync + Send + std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>
{
    ///*
    let r = a.par_iter().zip(b).zip(c).map(|((&x, &y), &z)| x * z + y);
    result.clear();
    result.par_extend(r);
    //*/
    /*
    let r = result.par_iter_mut().zip(a).zip(b).zip(c);

    r.for_each(|(((r, &x), &y), &z)| {
        unsafe {
            ptr::write(r as *mut T, x * z + y)
        };
    });
    */
}

pub trait SimdItem {
    type Elem: std::marker::Sized;

    fn store (self, array: &mut [Self::Elem], idx: usize) where Self: core::marker::Sized;
}

impl SimdItem for f64x4 {
    type Elem = f64;

    fn store (self, array: &mut [f64], idx: usize) {
        self.store(array, idx);
    }
}

impl SimdItem for f32x8 {
    type Elem = f32;

    fn store (self, array: &mut [f32], idx: usize) {
        self.store(array, idx);
    }
}

pub trait SimdCapable where Self: core::marker::Sized {
    type SimdType: SimdItem<Elem=Self> + core::fmt::Debug + std::marker::Copy + std::ops::Add<Output=Self::SimdType> + std::ops::Mul<Output=Self::SimdType>;
    const CHUNK_SIZE: usize;
    type ArrayType;

    fn load (array: &[Self], idx: usize) -> Self::SimdType where Self: core::marker::Sized;
}

impl SimdCapable for f64 {
    type SimdType = f64x4;
    type ArrayType = [f64; 4];
    const CHUNK_SIZE: usize = 4;

    fn load (array: &[Self], idx: usize) -> Self::SimdType {
        Self::SimdType::load(array, idx)
    }
}

impl SimdCapable for f32 {
    type SimdType = f32x8;
    type ArrayType = [f32; 8];
    const CHUNK_SIZE: usize = 8;

    fn load (array: &[Self], idx: usize) -> Self::SimdType {
        Self::SimdType::load(array, idx)
    }
}

pub fn vtriad_simd<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
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
            core::ptr::copy((&item as *const T::SimdType) as *const T, result_ptr.offset(index as isize), T::CHUNK_SIZE)
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

pub fn vtriad_simd_rayon<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
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

pub fn vtriad_threads<T>(result: &mut [T], a: &[T], b: &[T], c: &[T])
    where T: 'static + Sync + Send + std::marker::Copy + std::ops::Add<Output=T> + std::ops::Mul<Output=T>,
{
    let len = result.len();
    let num_cores = TOPOLOGY.objects_with_type(&ObjectType::PU).unwrap().len();

    assert_eq!(len % num_cores, 0);

    let part = len / num_cores;

    let mut res_iter = result.chunks_mut(part);
    let mut a_iter = a.chunks(part);
    let mut b_iter = b.chunks(part);
    let mut c_iter = c.chunks(part);

    let threads = (0..num_cores).map(|_| {
        let mut rc = unsafe {
            let slice = res_iter.next().unwrap();
            let len = slice.len();
            std::slice::from_raw_parts_mut(slice.as_mut_ptr(), len)
        };

        let ac = unsafe {
            let slice = a_iter.next().unwrap();
            let len = slice.len();
            std::slice::from_raw_parts(slice.as_ptr(), len)
        };
        let bc = unsafe {
            let slice = b_iter.next().unwrap();
            let len = slice.len();
            std::slice::from_raw_parts(slice.as_ptr(), len)
        };
        let cc = unsafe {
            let slice = c_iter.next().unwrap();
            let len = slice.len();
            std::slice::from_raw_parts(slice.as_ptr(), len)
        };

        std::thread::spawn(move || {
            vtriad_itertools_2(rc, ac, bc, cc);
        })
    }).collect::<Vec<_>>();

    #[allow(unused_must_use)]
    for thread in threads {
        thread.join();
    }
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

    #[test]
    fn test_vtriad_threads() {
        let mut result = vec![0.; 8];
        let a = [0., 1., 2., 3., 4., 5., 6., 7.];
        let b = [5., 4., 3., 2., 1., 0., 1., 2.];
        let c = [0., 1., 2., 3., 4., 5., 6., 7.];
        vtriad_threads(&mut result, &a, &b, &c);
        array_equal(&result, &[5., 5., 7., 11., 17., 25., 37., 51.]);
    }

    //#[bench]
    fn bench_copy(bencher: &mut test::Bencher) {
        let mut dst = random_array();
        let src = test::black_box(random_array());
        bencher.iter(|| {
            test::black_box(copy(&src, &mut dst));
        });
    }

    //#[bench]
    fn bench_add(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = test::black_box(random_array());
        let b = random_array();
        bencher.iter(|| {
            test::black_box(add(&mut res, &a, &b));
        });
    }

    //#[bench]
    fn bench_add_itertools(bencher: &mut test::Bencher) {
        let mut result = vec![0 as NumType; ARRAY_SIZE];
        let a = test::black_box(random_array());
        let b = random_array();
        bencher.iter(|| {
            test::black_box(add_itertools(&mut result, &a, &b));
        });
    }

    //#[bench]
    fn bench_striad(bencher: &mut test::Bencher) {
        let mut res = vec![0 as NumType; ARRAY_SIZE];
        let a = test::black_box(random_array());
        let b = random_array();
        bencher.iter(|| {
            test::black_box(striad(&mut res, &a, &b, S));
        });
    }

    //#[bench]
    fn bench_vtriad(bencher: &mut test::Bencher) {
        let (mut res, a, b, c) = prepare_arrays();

        bencher.iter(|| {
            test::black_box(vtriad(&mut res, &a, &b, &c));
        });
    }

    #[bench]
    fn bench_vtriad_itertools(bencher: &mut test::Bencher) {
        let (mut res, a, b, c) = prepare_arrays();

        //for i in 0..100 {
        bencher.iter(|| {
            vtriad_itertools(&mut res, &a, &b, &c);
        });
        //}
    }

    #[bench]
    fn bench_vtriad_itertools_2(bencher: &mut test::Bencher) {
        let (mut res, a, b, c) = prepare_arrays();

        bencher.iter(|| {
            test::black_box(vtriad_itertools_2(&mut res, &a, &b, &c));
        });
    }

    #[bench]
    fn bench_vtriad_rayon(bencher: &mut test::Bencher) {
        let (mut res, a, b, c) = prepare_arrays();

        bencher.iter(|| {
            test::black_box(vtriad_rayon(&mut res, &a, &b, &c));
        });
    }

    //#[bench]
    fn bench_vtriad_simd(bencher: &mut test::Bencher) {
        let (mut res, a, b, c) = prepare_arrays();

        bencher.iter(|| {
            test::black_box(vtriad_simd(&mut res, &a, &b, &c));
        });
    }

    //#[bench]
    fn bench_vtriad_simd_rayon(bencher: &mut test::Bencher) {
        let (mut res, a, b, c) = prepare_arrays();
        bencher.iter(|| {
            test::black_box(vtriad_simd_rayon(&mut res, &a, &b, &c));
        });
    }

    #[bench]
    fn bench_vtriad_threads(bencher: &mut test::Bencher) {
        let (mut res, a, b, c) = prepare_arrays();
        bencher.iter(|| {
            test::black_box(vtriad_threads(&mut res, &a, &b, &c));
        });
    }
}
