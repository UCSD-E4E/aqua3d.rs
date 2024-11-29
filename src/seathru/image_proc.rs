use std::{iter::Filter, ops::{Add, Div}};

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

pub trait TwoDimensionTransforms<A> {
    fn np_where(&self, val: A) -> Vec<(usize, usize)>;
}

impl<A> TwoDimensionTransforms<A> for Array2<A>
where
    A: Ord + Copy,
{
    fn np_where(&self, val: A) -> Vec<(usize, usize)> {
        self.indexed_iter()
            .filter(|&((_, _), &mat_val)| mat_val == val)
            .map(|(index, _)| index)
            .collect()
    }
}

impl<T: Add<Output = T>> Add for (T, T) {
    type Output = (T, T);

    fn add(self, other: Self) -> Self::Output {
        (self.0 + other.0, self.1 + other.1)
    }
}