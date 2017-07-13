
extern crate rand;

extern crate simd;
use self::simd::x86::avx::f64x4;

use std;

use std::ops::{Index, IndexMut};

use std::cmp::{PartialEq, Eq};

#[derive(Debug, PartialEq)]
pub struct Matrix {
    pub data: Vec<f64>, // force heap allocation
    pub rows: usize,
    pub columns: usize,
    aligned: bool,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize, data: Vec<f64>) -> Self {
        Self {
            rows: rows,
            columns: columns,
            data: data,
            aligned: false,
        }
    }

    pub fn new_aligned(rows: usize, columns: usize, data: Vec<f64x4>) -> Self {

        let vec = unsafe {
            let ret = Vec::from_raw_parts(data.as_ptr() as *mut f64, data.len() * 4, data.capacity() * 4);
            std::mem::forget(data);
            ret
        };

        Self {
            rows: rows,
            columns: columns,
            data: vec,
            aligned: true,
        }
    }

    ///
    ///
    pub fn zero(rows: usize, columns: usize) -> Self {
        if columns % 4 == 0 {
            Self::new_aligned(rows, columns, vec![f64x4::splat(0.); rows * columns / 4])
        }
        else {
            Self::new(rows, columns, vec![0 as f64; rows * columns])
        }
    }

    pub fn random(rows: usize, columns: usize) -> Self {
        if columns % 4 == 0 {
            Self::new_aligned(rows, columns, (0..rows * columns / 4).map(|i| f64x4::splat(i as f64)).collect())
        }
        else {
            Self::new(rows, columns, (0..rows*columns).map(|i| i as f64).collect())
        }
    }

    pub fn reset(&mut self) {
        for v in &mut self.data {
            *v = 0.;
        }
    }

    pub fn is_aligned(&self) -> bool {
        self.aligned
    }
}

pub trait IndexUnchecked<T> : Index<T> {
    #[inline]
    unsafe fn get_unchecked(&self, index: T) -> &Self::Output;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: T) -> &mut Self::Output;
}

impl IndexUnchecked<isize> for Matrix {
    #[inline]
    unsafe fn get_unchecked(&self, index: isize) -> &f64 {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: isize) -> &mut f64 {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<usize> for Matrix {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &f64 {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut f64 {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<(usize, usize)> for Matrix {
    #[inline]
    unsafe fn get_unchecked(&self, (row, column): (usize, usize)) -> &f64 {
        let index = row * self.columns + column;
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, (row, column): (usize, usize)) -> &mut f64 {
        let index = row * self.columns + column;
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl Eq for Matrix {}

impl Index<isize> for Matrix {
    type Output = f64;

    #[inline]
    fn index(&self, index: isize) -> &f64 {
        &self.data[index as usize]
    }
}

impl Index<i32> for Matrix {
    type Output = f64;

    #[inline]
    fn index(&self, index: i32) -> &f64 {
        &self.data[index as usize]
    }
}

impl Index<usize> for Matrix {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &f64 {
        &self.data[index as usize]
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    #[inline]
    fn index(&self, (row, column): (usize, usize)) -> &f64 {
        &self.data[row * self.columns + column]
    }
}

impl IndexMut<isize> for Matrix {
   #[inline]
    fn index_mut(&mut self, index: isize) -> &mut f64 {
        &mut self.data[index as usize]
    }
}

impl IndexMut<usize> for Matrix {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.data[index as usize]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut f64 {
        &mut self.data[row * self.columns + column]
    }
}