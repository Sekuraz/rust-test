
extern crate rand;

use std::ops::{Index, IndexMut};

use std::cmp::{PartialEq, Eq};

type DataType = f64;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    pub data: Vec<DataType>, // force heap allocation
    pub rows: usize,
    pub columns: usize,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize, data: Vec<DataType>) -> Self {
        Self {
            rows: rows,
            columns: columns,
            data: data,
        }
    }
    pub fn random(rows: usize, columns: usize) -> Self {
        Self {
            rows: rows,
            columns: columns,
            data: (0..rows*columns).map(|i| i as DataType).collect(),
        }
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
    unsafe fn get_unchecked(&self, index: isize) -> &DataType {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: isize) -> &mut DataType {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<usize> for Matrix {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &DataType {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut DataType {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<(usize, usize)> for Matrix {
    #[inline]
    unsafe fn get_unchecked(&self, (row, column): (usize, usize)) -> &DataType {
        let index = row * self.columns + column;
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, (row, column): (usize, usize)) -> &mut DataType {
        let index = row * self.columns + column;
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl Eq for Matrix {}

impl Index<isize> for Matrix {
    type Output = DataType;

    #[inline]
    fn index(&self, index: isize) -> &DataType {
        &self.data[index as usize]
    }
}

impl Index<i32> for Matrix {
    type Output = DataType;

    #[inline]
    fn index(&self, index: i32) -> &DataType {
        &self.data[index as usize]
    }
}

impl Index<usize> for Matrix {
    type Output = DataType;

    #[inline]
    fn index(&self, index: usize) -> &DataType {
        &self.data[index as usize]
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = DataType;

    #[inline]
    fn index(&self, (row, column): (usize, usize)) -> &DataType {
        &self.data[row * self.columns + column]
    }
}

impl IndexMut<isize> for Matrix {
   #[inline]
    fn index_mut(&mut self, index: isize) -> &mut DataType {
        &mut self.data[index as usize]
    }
}

impl IndexMut<usize> for Matrix {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut DataType {
        &mut self.data[index as usize]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut DataType {
        &mut self.data[row * self.columns + column]
    }
}