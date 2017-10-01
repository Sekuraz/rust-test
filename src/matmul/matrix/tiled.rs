
use super::standard::Matrix;

use std;

use std::ops::{Index, IndexMut};

use std::cmp::{PartialEq, Eq};

use std::convert::From;

use std::fmt;


/// A rust tiled Matrix
///
/// This is a row major matrix!
///
/// # Properties
/// ## Members
/// This matrix consists of the following members:
///
/// rows:           usize       The number of rows
/// columns:        usize       The number of columns
/// blocks_right:   usize       The number of blocks in the first dimension. The last block might be padded.
/// blocks_down:    usize       The number of blocks in the second dimension. The last block might be padded.
/// data:           Vec<Matrix> The matrix elements stored in this matrix
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

#[derive(Debug, PartialEq, Clone)]
pub struct TileMatrix {
    pub rows: usize,
    pub columns: usize,
    pub blocks_right: usize,
    pub blocks_down: usize,
    pub data: Vec<Matrix>, // force heap allocation
    block_rows: usize,
    block_columns: usize,
}

impl TileMatrix {
    const BLOCK_SIZE: usize = 128; // best for naive_simd algorithm

    /// Create a new tiled matrix from the given parts
    pub fn new(blocks_right: usize, blocks_down: usize, data: Vec<Matrix>) -> Self {
        assert_eq!(data.len(), blocks_right * blocks_down);
        let r = data[0].rows;
        let c = data[0].columns;

        for matrix in &data {
            assert_eq!(r, matrix.rows);
            assert_eq!(c, matrix.columns);
        }


        Self {
            rows: r * blocks_down,
            columns: c * blocks_right,
            blocks_right: blocks_right,
            blocks_down: blocks_down,
            data: data,
            block_rows: r,
            block_columns: c,
        }
    }

    pub fn new_with_size(rows: usize, columns: usize, blocks_right: usize, blocks_down: usize, data: Vec<Matrix>) -> Self {
        let mut ret = Self::new(blocks_right, blocks_down, data);
        ret.rows = rows;
        ret.columns = columns;
        ret
    }

    /// Create a zero matrix
    ///
    /// This matrix is aligned if columns is a multiple of 4 * blocks_right.
    ///
    pub fn zero(rows: usize, columns: usize, blocks_right: usize, blocks_down: usize) -> Self {

        let r = if rows % blocks_down == 0 { rows / blocks_down } else { rows / blocks_down + 1 };
        let c = if columns % blocks_right == 0 { columns / blocks_right } else { columns / blocks_right + 1 };
        let data = (0..blocks_right * blocks_down).map(|_| {
            Matrix::zero(r, c)
        }).collect::<Vec<_>>();

        Self::new_with_size(rows, columns, blocks_right, blocks_down, data)
    }

    /// Create a "random" matrix
    ///
    /// This matrix is aligned if columns is a multiple of 4.
    /// The data itself is not random but 0..rows*columns if not aligned and 0..rows * columns / 4
    /// with each value occurring 4 times if aligned
    ///
    pub fn random(rows: usize, columns: usize, blocks_right: usize, blocks_down: usize) -> Self {

        let r = if rows % blocks_down == 0 { rows / blocks_down } else { rows / blocks_down + 1 };
        let c = if columns % blocks_right == 0 { columns / blocks_right } else { columns / blocks_right + 1 };
        let data = (0..blocks_right * blocks_down).map(|_| {
            Matrix::random(r, c)
        }).collect::<Vec<_>>();

        Self::new_with_size(rows, columns, blocks_right, blocks_down, data)
    }

    /// Reset all entries to zero
    pub fn reset(&mut self) {
        for ref mut matrix in &mut self.data {
            matrix.reset();
        }
    }

    /// Check whether this matrix is aligned and can be used in simd pointer casts
    pub fn is_aligned(&self) -> bool {
        self.data.iter().all(|matrix| matrix.is_aligned())
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

impl Eq for TileMatrix {}

impl Index<isize> for TileMatrix {
    type Output = Matrix;

    #[inline]
    fn index(&self, index: isize) -> &Matrix {
        &self.data[index as usize]
    }
}

impl Index<i32> for TileMatrix {
    type Output = Matrix;

    #[inline]
    fn index(&self, index: i32) -> &Matrix {
        &self.data[index as usize]
    }
}

impl Index<usize> for TileMatrix {
    type Output = Matrix;

    #[inline]
    fn index(&self, index: usize) -> &Matrix {
        &self.data[index as usize]
    }
}

impl Index<(usize, usize)> for TileMatrix {
    type Output = Matrix;

    #[inline]
    fn index(&self, (row, column): (usize, usize)) -> &Matrix {
        &self.data[row * self.blocks_right + column]
    }
}


impl IndexMut<isize> for TileMatrix {
   #[inline]
    fn index_mut(&mut self, index: isize) -> &mut Matrix {
        &mut self.data[index as usize]
    }
}

impl IndexMut<usize> for TileMatrix {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Matrix {
        &mut self.data[index as usize]
    }
}

impl IndexMut<(usize, usize)> for TileMatrix {
    #[inline]
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut Matrix {
        &mut self.data[row * self.blocks_right + column]
    }
}

impl From<Matrix> for TileMatrix {
    fn from(matrix: Matrix) -> Self {
        if matrix.rows < Self::BLOCK_SIZE || matrix.columns < Self::BLOCK_SIZE {
            Self::new(1,1, vec![matrix])
        }
        else {
            let rows = matrix.rows;
            let columns = matrix.columns;
            let bs = Self::BLOCK_SIZE;

            let blocks_down = if rows % bs == 0 { rows / bs } else { rows / bs + 1 };
            let blocks_right = if columns % bs == 0 { columns / bs } else { columns / bs + 1 };

            //let chunks = vec![vec![]; blocks_right];

            let mut vec_data = vec![vec![]; blocks_right * blocks_down];

            for (row_ind, row) in matrix.data.chunks(columns).enumerate() {
                for (chunk_ind, chunk) in row.chunks(bs).enumerate() {
                    vec_data[(row_ind / bs) * blocks_right + chunk_ind].extend_from_slice(chunk);
                }
            }

            let mut data = Vec::new();

            for (i, mat) in vec_data.iter().enumerate() {
                let mut new_mat = Matrix::zero(bs, bs);

                if (i + 1) % blocks_right == 0 && columns % bs != 0 {
                    let row_len = rows % bs;
                    new_mat.data.chunks_mut(bs).zip(mat.chunks(row_len)).for_each(
                        |(row, row_data)| row.iter_mut().zip(row_data).for_each(
                            |(new, d)| *new = *d
                        )
                    );
                }
                else {
                    if mat.len() == bs * bs {
                        new_mat.data.copy_from_slice(mat);
                    }
                    else {
                        new_mat.data.iter_mut().zip(mat).for_each(|(new, d)| *new = *d);
                    }
                }
                data.push(new_mat)
            }

            Self::new_with_size(rows, columns, blocks_right, blocks_down, data)
        }
    }
}

impl fmt::Display for TileMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", std::iter::repeat("-").take(self.columns * 10 + 2 * self.blocks_right + 1 ).collect::<String>())?;

        for c_row in 0..self.blocks_down {
            for b_row in 0..self.block_rows {
                write!(f, "| ")?;
                for c_col in 0..self.blocks_right {
                    for b_col in 0..self.block_columns {
                        let entry = &self[(c_row, c_col)][(b_row, b_col)];
                        write!(f, "{:>8}, ", entry)?;
                    }
                    write!(f, "| ")?;
                }
                writeln!(f)?;
            }
            writeln!(f, "{}", std::iter::repeat("-").take(self.columns * 10 + 2 * self.blocks_right + 1 ).collect::<String>())?;
        }
        Ok(())
    }
}