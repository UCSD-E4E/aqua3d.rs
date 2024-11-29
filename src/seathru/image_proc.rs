use std::{iter::Filter, ops::Div};

use ndarray::iter::IndexedIter;
/// image_proc.rs
/// Author  : Avik Ghosh
/// Purpose : Temporarily add ndarray methods that are frequently reused in porting seathru
///             Will be moved to an overall Image-Processing library that will be used by more of
///             FS/A3D Code
///             Original repo: https://github.com/hainh/sea-thru
///           To be used by FishSense as library for data processing.

// Imports
// use num_traits::ToPrimitive;
// use ordered_float::OrderedFloat;
// use ndarray_stats::QuantileExt;
#[allow(unused)]
use ndarray::{array, Array, Array1, Array2, Array3, Axis};

// use std::{cmp::min, collections::VecDeque};
// use itertools::Itertools;

use anyhow::{anyhow, Context, Result};
pub trait ImageProcessing<A> {
    fn grayscale(&self) -> Array2<f32>;
    // fn np_where(&self, predicate: P) -> Filter<Self, P>;
}

impl<A> ImageProcessing<A> for Array3<A>
where
    A: Ord + Copy + Into<f32>, // + Into<B>, // B : From<Div<i32>>  + std::ops::Div,
{
    fn grayscale(&self) -> Array2<f32> {
        self.map(|val| (Into::<f32>::into(*val) / 3.0))
            .sum_axis(Axis(2))
    }
}
