
use std::ops::{Index, IndexMut};

pub trait IndexUnchecked<T> : Index<T> {
    #[inline]
    unsafe fn get_unchecked(&self, index: T) -> &Self::Output;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: T) -> &mut Self::Output;
}