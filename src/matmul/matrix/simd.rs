
extern crate rand;

extern crate simd;
use self::simd::x86::avx::f64x4;

use std;

use std::fmt;

use std::ops::{Index, IndexMut, Deref};

use std::cmp::{PartialEq, Eq};

use std::convert::From;

use super::traits::*;
use super::Matrix;

/// A rust Matrix containing f64x4
///
/// This is a row major matrix!
///
/// # Properties
/// ## Members
/// This matrix consists of the following members:
///
/// rows:       usize       The number of rows
/// columns:    usize       The number of columns
/// data:       Vec<f64x4>    The data stored in this matrix
///
/// data is allocated on the heap because rust has a limit of 2MB on its stack.
///
/// ## Alignment
/// If the matrix reports alignment via is_aligned(), data's content is aligned to 32 bytes and can
/// be used as the f64x4 by simple pointer casting.
///
/// ## Methods
/// This struct only implements methods to create, alter and index itself. Operations have to be
/// implemented somewhere else.
///
#[derive(Debug, Clone)]
pub struct SimdMatrix {
    pub rows: usize,
    pub columns: usize,
    aligned: bool,
    pub data: Vec<f64x4>, // force heap allocation
}

impl SimdMatrix {
    /// Create a new, unaligned matrix from the given parts
    pub fn new(rows: usize, columns: usize, data: Vec<f64x4>) -> Self {
        Self {
            rows: rows,
            columns: columns,
            data: data,
            aligned: true,
        }
    }

    /// Reset all entries to zero
    pub fn reset(&mut self) {
        let zero = f64x4::splat(0.0);
        for v in &mut self.data {
            *v = zero;
        }
    }

    /// Check whether this matrix is aligned and can be used in simd pointer casts
    pub fn is_aligned(&self) -> bool {
        self.aligned
    }
}


impl IndexUnchecked<isize> for SimdMatrix {
    #[inline]
    unsafe fn get_unchecked(&self, index: isize) -> &f64x4 {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: isize) -> &mut f64x4 {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<usize> for SimdMatrix {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &f64x4 {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut f64x4 {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<(usize, usize)> for SimdMatrix {
    #[inline]
    unsafe fn get_unchecked(&self, (row, column): (usize, usize)) -> &f64x4 {
        let index = row * self.columns + column;
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, (row, column): (usize, usize)) -> &mut f64x4 {
        let index = row * self.columns + column;
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl Eq for SimdMatrix {}

impl PartialEq<SimdMatrix> for SimdMatrix {
    fn eq(&self, other: &SimdMatrix) -> bool {
        self.columns == other.columns &&
            self.rows == other.rows //&&
            //self.data.iter().zip(other.data).all(|(lhs, rhs)| (lhs.eq(rhs)).all())
    }
}

impl Index<isize> for SimdMatrix {
    type Output = f64x4;

    #[inline]
    fn index(&self, index: isize) -> &f64x4 {
        &self.data[index as usize]
    }
}

impl Index<i32> for SimdMatrix {
    type Output = f64x4;

    #[inline]
    fn index(&self, index: i32) -> &f64x4 {
        &self.data[index as usize]
    }
}

impl Index<usize> for SimdMatrix {
    type Output = f64x4;

    #[inline]
    fn index(&self, index: usize) -> &f64x4 {
        &self.data[index as usize]
    }
}

impl Index<(usize, usize)> for SimdMatrix {
    type Output = f64x4;

    #[inline]
    fn index(&self, (row, column): (usize, usize)) -> &f64x4 {
        &self.data[row * self.columns + column]
    }
}

impl IndexMut<isize> for SimdMatrix {
   #[inline]
    fn index_mut(&mut self, index: isize) -> &mut f64x4 {
        &mut self.data[index as usize]
    }
}

impl IndexMut<usize> for SimdMatrix {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f64x4 {
        &mut self.data[index as usize]
    }
}

impl IndexMut<(usize, usize)> for SimdMatrix {
    #[inline]
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut f64x4 {
        &mut self.data[row * self.columns + column]
    }
}

impl From<Matrix> for SimdMatrix {
    fn from(matrix: Matrix) -> Self {
        assert!(matrix.is_aligned());

         let vec = unsafe {
             let ret = Vec::from_raw_parts(matrix.data.as_ptr() as *mut f64x4, matrix.data.len() / 4, matrix.data.capacity() / 4);
             std::mem::forget(matrix.data);
             ret
         };

        Self::new(matrix.rows, matrix.columns / 4, vec)
    }
}

impl fmt::Display for SimdMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", std::iter::repeat("-").take(self.columns * 14 + 3).collect::<String>())?;
        self.data.chunks(self.columns).for_each(|row| {
            write!(f, "| ");
            for item in row {
                write!(f, "{:>48?}, ", item);
            }
            writeln!(f, "|");
        });
        writeln!(f, "{}", std::iter::repeat("-").take(self.columns * 14 + 3).collect::<String>())

    }
}
