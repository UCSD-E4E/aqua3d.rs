mod image_proc;

use image_proc::{ImageProcessing, TwoDimensionTransforms};
use ndarray::{array, s, Array, Array1, Array2, Array3};
use ndarray_stats::QuantileExt;
use num_traits::ToPrimitive;
use ordered_float::OrderedFloat;
/// seathru.rs
/// Author  : Avik Ghosh
/// Purpose : Migrate individual functions from sea-thru, a library used to correct for
///             wavelength dependent attenuation of light in underwater images.
///             Original repo: https://github.com/hainh/sea-thru
///           To be used by FishSense as library for data processing.

// Includes
#[allow(unused_imports)]
use rand::Rng;

use itertools::Itertools;
use std::{cmp::min, collections::VecDeque, ops::RangeBounds};

use anyhow::{anyhow, Context, Result};

#[allow(non_snake_case)]
#[allow(dead_code)]
pub struct RgbdData<'a> {
    pub img: &'a Array3<u8>,
    pub depth_map: &'a Array2<f32>,
    pub R: usize, // RGB/BGR Indices. used in find_backscatter_esimation_points
    pub G: usize,
    pub B: usize,
}

pub trait Seathru {
    #[allow(dead_code)]
    fn run_pipeline(&self, is_rand: bool) -> Result<()>;
    fn construct_neighborhood_map(&self, eps: f32, is_rand: bool) -> Result<(Array2<i32>, i32)>;

    #[allow(dead_code)]
    fn find_backscatter_estimation_points(
        &self,
        num_bins: usize,
        fraction: f32,
        max_vals: u32,
        min_depth_percentage: f32,
    ) -> Result<Vec<Array1<(f32, u8)>>>;
}

impl<'a> Seathru for RgbdData<'a> {
    fn run_pipeline(&self, is_rand: bool) -> Result<()> {
        #[allow(unused)]
        let estimation_pts = self
            .find_backscatter_estimation_points(10, 0.05, 20, 0.3)
            .context("Failed to set up backscatter estimation points")?;

        #[allow(unused)]
        // Don't want to add underscores as we're incrementally building. Remove later
        let (nmap, _num_neighborhoods) = self
            .construct_neighborhood_map(1e-5, is_rand)
            .context("Failed to set up neighborhood map")?;

        Ok(())
    }

    #[allow(unused)]
    fn find_backscatter_estimation_points(
        &self,
        num_bins: usize,
        fraction: f32,
        max_vals: u32,
        min_depth_percentage: f32,
    ) -> Result<Vec<Array1<(f32, u8)>>> {
        // Return vectors
        let mut points: Vec<Array1<(f32, u8)>> = vec![
            Array1::from_elem(max_vals as usize, (0 as f32, 0 as u8)), // First empty array
            Array1::from_elem(max_vals as usize, (0 as f32, 0 as u8)), // Second empty array
            Array1::from_elem(max_vals as usize, (0 as f32, 0 as u8)), // Third empty array
        ];

        // Rename all of these bc what were they thinking???
        let (min_depth, max_depth) = (*self.depth_map.min()?, *self.depth_map.max()?);
        let min_depth_thresh = min_depth + (min_depth_percentage * (max_depth - min_depth));
        let depth_ranges = Array1::linspace(min_depth, max_depth, num_bins + 1);

        let img_norms: Array2<f32> = self.img.grayscale();
        let mut num_pts_sent = 0;

        for i in 0..depth_ranges.len() - 1 {
            let location_closure = |&((_i, _j), &value): &((usize, usize), &f32)| {
                value > min_depth_thresh && value >= depth_ranges[i] && value <= depth_ranges[i + 1]
            };

            let locs = self.depth_map.indexed_iter().filter(location_closure);

            // TODO: Find more efficient way to do this without cloning. GPT says to make a filter condition and reuse that instead?
            let norms = locs.clone().map(|(index, _)| img_norms[index]);
            let depths = locs.clone().map(|(_, val)| *val);
            let pixels = locs.map(|((x, y), _)| self.img.slice(s![x, y, ..]).to_owned());

            // Data is each of the last vectors zipped together
            let data: Vec<(f32, Array1<u8>)> = norms
                .zip(pixels)
                .zip(depths)
                .sorted_by_key(|((norm, pixel), depth)| OrderedFloat(*norm))
                .map(|((norm, pixel), depth)| (depth, pixel))
                .collect();

            // TODO: Find how to make sure these integer conversions are safer and what gets rounded/cut off/put to max int val, etc.
            // Also, this is currently a Python -> Rust conversion but why would they do an iterator starting at 1 and not 0? Starting 0 here
            let pts_to_send = num_pts_sent..min(
                (fraction * data.len() as f32).ceil() as usize,
                max_vals as usize,
            );
            
            // Increment num_pts_sent
            num_pts_sent = pts_to_send.end + 1; 

            for next_pt_ind in pts_to_send {
                let data_pt = data.get(next_pt_ind).ok_or_else(|| {
                    anyhow!("Could not properly access estimated backscatter points")
                })?;
                points[self.R][next_pt_ind] = (data_pt.0, data_pt.1[self.R]);
                points[self.G][next_pt_ind] = (data_pt.0, data_pt.1[self.G]);
                points[self.B][next_pt_ind] = (data_pt.0, data_pt.1[self.B]);
            }
        }

        Ok(points)
    }

    fn construct_neighborhood_map(&self, eps: f32, is_rand: bool) -> Result<(Array2<i32>, i32)> {
        let scaled_eps = (*self.depth_map.max()? - *self.depth_map.min()?) * eps;
        let mut neighborhood_map: Array2<i32> = Array::zeros(self.depth_map.raw_dim()); // Convert to u32 soon
        let mut num_neighborhoods = 1;

        // Discretization loop
        // Condition is simple way to make sure the boolean map of zero values isn't empty
        while *neighborhood_map.min()? == 0 {
            let zero_map: Vec<(usize, usize)> = neighborhood_map.np_where(0);

            let (start_x, start_y) = match is_rand {
                true => zero_map[rand::thread_rng().gen_range(0..zero_map.len())],
                false => zero_map[0 as usize],
            };

            let mut q = VecDeque::<(usize, usize)>::new();
            q.push_back((start_x, start_y));

            let (mut x_ind, mut y_ind);

            while q.len() != 0 {
                (x_ind, y_ind) = q.pop_front().context("Empty queue")?;

                if (self.depth_map[(x_ind, y_ind)] - self.depth_map[(start_x, start_y)]).abs()
                    <= scaled_eps
                {
                    neighborhood_map[(x_ind, y_ind)] = num_neighborhoods;

                    // Add in neighboring points. x_ind and y_ind are already positive or 0, so no need to check that.
                    // Array of arrays for all Deltas from (x_ind, y_ind) to have the 4 surrounding (non-diagonal) points.
                    let nearby_inds = array![(-1, 0), (1, 0), (0, -1), (0, 1)]
                        .map(|del| (del.0 + x_ind as i32, del.1 + y_ind as i32));

                    for (del_x_ind, del_y_ind) in nearby_inds {
                        if let (Some(x), Some(y)) = (del_x_ind.to_usize(), del_y_ind.to_usize()) {
                            if let Some(val) = neighborhood_map.get((x, y)) {
                                if *val == 0 {
                                    q.push_back((x, y));
                                }
                            }
                        }
                    }
                }
            }

            // Increment
            num_neighborhoods += 1;
        }

        // TODO: Why are we doing equality on floating point?? is depth integer then? Currently using the below value
        // Still can't think of any real reason to avoid modifying this to num_neighborhood.map(|x| x - 1) so it starts at 0.
        let practically_zero = *self.depth_map.min()? + 1e-5;

        // Boolean map for depth indices that are 'practically zero', get vector of indices whose depths are
        // 'practically zero', then return iterator of the neighborhood_map values at those indices, collect() on iterator.
        let zero_depth_nmap: Vec<i32> = (self.depth_map)
            .map(|x| *x < practically_zero)
            .indexed_iter()
            .filter(|&((_, _), &is_zero)| is_zero)
            .map(|((x, y), _)| neighborhood_map[(x, y)])
            .collect();

        // Most common nmap value for 0 depth can be found in counts
        let counts = zero_depth_nmap.iter().counts();
        let (most_frequent, _) = counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .context("Empty depth matrix")?;

        // TODO: Figure out why we want a gap in the neighborhood mapping.
        Ok((
            neighborhood_map.map(|&nmap_val| {
                if nmap_val == *most_frequent {
                    0
                } else {
                    nmap_val
                }
            }),
            num_neighborhoods - 1,
        ))
    }
}
