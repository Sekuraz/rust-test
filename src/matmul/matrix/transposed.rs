
extern crate rand;

extern crate simd;
use self::simd::x86::avx::f64x4;

use std;

use std::fmt;

use std::ops::{Index, IndexMut};

use std::cmp::{PartialEq, Eq};

use super::traits::*;

use super::standard::Matrix;

/// A rust Matrix
///
/// This is a column major matrix!
///
/// # Properties
/// ## Members
/// This matrix consists of the following members:
///
/// rows:       usize       The number of rows
/// columns:    usize       The number of columns
/// data:       Vec<f64>    The data stored in this matrix
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
#[derive(Debug, PartialEq, Clone)]
pub struct TransposedMatrix {
    pub rows: usize,
    pub columns: usize,
    aligned: bool,
    pub data: Vec<f64>, // force heap allocation
}

impl TransposedMatrix {
    /// Create a new, unaligned matrix from the given parts
    pub fn new(rows: usize, columns: usize, data: Vec<f64>) -> Self {
        Self {
            rows: rows,
            columns: columns,
            data: data,
            aligned: false,
        }
    }

    /// Create a new, aligned matrix
    ///
    /// data is of type Vec<f64x4> in order to ensure it is correctly aligned and is casted to
    /// Vec<f64> during construction.
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

    /// Create a zero matrix
    ///
    /// This matrix is aligned if columns is a multiple of 4.
    ///
    pub fn zero(rows: usize, columns: usize) -> Self {
        if columns % 4 == 0 {
            Self::new_aligned(rows, columns, vec![f64x4::splat(0.); rows * columns / 4])
        }
        else {
            Self::new(rows, columns, vec![0 as f64; rows * columns])
        }
    }

    /// Create a "random" matrix
    ///
    /// This matrix is aligned if columns is a multiple of 4.
    /// The data itself is not random but 0..rows*columns if not aligned and 0..rows * columns / 4
    /// with each value occurring 4 times if aligned
    ///
    pub fn random(rows: usize, columns: usize) -> Self {
        if columns % 4 == 0 {
            Self::new_aligned(rows, columns, (0..rows * columns / 4).map(|i| f64x4::splat(i as f64)).collect())
        }
        else {
            Self::new(rows, columns, (0..rows*columns).map(|i| i as f64).collect())
        }
    }

    /// Reset all entries to zero
    pub fn reset(&mut self) {
        for v in &mut self.data {
            *v = 0.;
        }
    }

    /// Check whether this matrix is aligned and can be used in simd pointer casts
    pub fn is_aligned(&self) -> bool {
        self.aligned
    }
}


impl IndexUnchecked<isize> for TransposedMatrix {
    #[inline]
    unsafe fn get_unchecked(&self, index: isize) -> &f64 {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: isize) -> &mut f64 {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<usize> for TransposedMatrix {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &f64 {
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut f64 {
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl IndexUnchecked<(usize, usize)> for TransposedMatrix {
    #[inline]
    unsafe fn get_unchecked(&self, (row, column): (usize, usize)) -> &f64 {
        let index = column * self.rows + row;
        &*self.data.as_ptr().offset(index as isize)
    }

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, (row, column): (usize, usize)) -> &mut f64 {
        let index = row * self.columns + column;
        &mut *self.data.as_mut_ptr().offset(index as isize)
    }
}

impl Eq for TransposedMatrix {}

impl Index<isize> for TransposedMatrix {
    type Output = f64;

    #[inline]
    fn index(&self, index: isize) -> &f64 {
        &self.data[index as usize]
    }
}

impl Index<i32> for TransposedMatrix {
    type Output = f64;

    #[inline]
    fn index(&self, index: i32) -> &f64 {
        &self.data[index as usize]
    }
}

impl Index<usize> for TransposedMatrix {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &f64 {
        &self.data[index as usize]
    }
}

impl Index<(usize, usize)> for TransposedMatrix {
    type Output = f64;

    #[inline]
    fn index(&self, (row, column): (usize, usize)) -> &f64 {
        &self.data[column * self.rows + row]
    }
}

impl IndexMut<isize> for TransposedMatrix {
   #[inline]
    fn index_mut(&mut self, index: isize) -> &mut f64 {
        &mut self.data[index as usize]
    }
}

impl IndexMut<usize> for TransposedMatrix {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.data[index as usize]
    }
}

impl IndexMut<(usize, usize)> for TransposedMatrix {
    #[inline]
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut f64 {
        &mut self.data[column * self.rows + row]
    }
}

impl From<Matrix> for TransposedMatrix {
    fn from(matrix: Matrix) -> Self {
        let mut tmp = Matrix::zero(matrix.columns, matrix.rows);

        for i in 0..matrix.rows {
            for j in 0..matrix.columns {
                tmp[(j, i)] = matrix[(i,j)];
            }
        }

        Self {
            rows: matrix.rows,
            columns: matrix.columns,
            aligned: tmp.is_aligned(),
            data: tmp.data,
        }
    }
}

impl fmt::Display for TransposedMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", std::iter::repeat("-").take(self.columns * 14 + 3).collect::<String>())?;
        self.data.chunks(self.columns).for_each(|row| {
            write!(f, "| ");
            for item in row {
                write!(f, "{:>12}, ", item);
            }
            writeln!(f, "|");
        });
        writeln!(f, "{}", std::iter::repeat("-").take(self.columns * 14 + 3).collect::<String>())
    }
}