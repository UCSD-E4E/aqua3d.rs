/// seathru.rs
/// Author  : Avik Ghosh
/// Purpose : Migrate individual functions from sea-thru, a library used to correct for 
///             wavelength dependent attenuation of light in underwater images.
///             Original repo: https://github.com/hainh/sea-thru
///           To be used by FishSense as library for data processing.

// Includes
#[allow(unused_imports)]
use rand::Rng;
use num_traits::ToPrimitive;
use ndarray_stats::QuantileExt;
use ndarray::{array, Array, Array2};

use std::collections::VecDeque;
use itertools::Itertools;

use anyhow::{Context, Result};

pub struct RgbdData<'a> {
    #[allow(dead_code)]
    pub img : &'a Array2<u8>,
    pub depth_map : &'a Array2<f32>,
}

pub trait Seathru {
    #[allow(dead_code)]    
    fn run_pipeline(&self, is_rand : bool) -> Result<()>;
    fn construct_neighborhood_map(&self, eps: f32, is_rand : bool) -> Result<(Array2<i32>, i32)> ;
}

impl<'a> Seathru for RgbdData<'a> {    
    fn run_pipeline(&self, is_rand : bool) -> Result<()> {
        #[allow(unused)]        // Don't want to add underscores as we're incrementally building. Remove later
        let (nmap, _num_neighborhoods) = self.construct_neighborhood_map(1e-5, is_rand)
                                                                                .context("Failed to set up neighborhood map")?;
        Ok(())  
    }

    fn construct_neighborhood_map(&self, eps: f32, is_rand : bool)  -> Result<(Array2<i32>, i32)> {
        let scaled_eps = (*self.depth_map.max()? - *self.depth_map.min()?) * eps;
        let mut neighborhood_map : Array2<i32> = Array::zeros(self.depth_map.raw_dim());         // Convert to u32 soon
        let mut num_neighborhoods = 1;

        // Discretization loop
        // Condition is simple way to make sure the boolean map of zero values isn't empty
        while *neighborhood_map.min()? == 0 {           
            let zero_map : Vec<(usize, usize)> = neighborhood_map.indexed_iter()
                                                                    .filter(|&((_, _), &value)| value == 0)
                                                                    .map(|(index, _)| index)
                                                                    .collect();

            let (start_x, start_y) = match is_rand {
                true => zero_map[rand::thread_rng().gen_range(0..zero_map.len())],
                false => zero_map[0 as usize]
            };

            let mut q = VecDeque::<(usize, usize)>::new();
            q.push_back((start_x, start_y));

            let mut x_ind : usize;
            let mut y_ind : usize;

            while q.len() != 0 {
                (x_ind, y_ind) = q.pop_front().context("Empty queue")?;

                if (self.depth_map[(x_ind, y_ind)] - self.depth_map[(start_x, start_y)]).abs() <= scaled_eps {
                    neighborhood_map[(x_ind, y_ind)] = num_neighborhoods;

                    // Add in neighboring points. x_ind and y_ind are already positive or 0, so no need to check that.
                    // Array of arrays for all Deltas from (x_ind, y_ind) to have the 4 surrounding (non-diagonal) points.
                    let deltas = array![(-1, 0), (1, 0), (0, -1), (0, 1)]
                                                                                .map(|x| array![x.0 as i32, x.1 as i32]);

                    for delta in deltas {
                        let surr_inds = &array![x_ind as i32, y_ind as i32] + &delta;

                        let [del_x_ind, del_y_ind] = *surr_inds.as_slice().expect("Index should size 2") else {
                            todo!("Please handle the case where the index is not of size 2!");
                        };

                        if let (Some(x), Some(y)) = (del_x_ind.to_usize(), del_y_ind.to_usize()) {
                            if let Some(val) = neighborhood_map.get((x, y)) {
                                if *val == 0 { q.push_back((x, y)); }
                            }
                        }
                    }
                }
            }

            // Increment
            num_neighborhoods += 1;
        }

        // TODO: Why are we doing equality on floating point?? is depth integer then? Currently using 
        let practically_zero = *self.depth_map.min()? + 1e-5;

        // Boolean map for depth indices that are 'practically zero', get vector of indices whose depths are 
        // 'practically zero', then return iterator of the neighborhood_map values at those indices, collect() on iterator.
        let zero_depth_nmap : Vec<i32> = self.depth_map.map(|x| *x < practically_zero)
                                                                        .indexed_iter()
                                                                        .filter(|&((_, _), &is_zero)| is_zero)
                                                                        .map(|((x, y), _)| neighborhood_map[(x, y)])
                                                                        .collect();


        // Most common nmap value for 0 depth can be found in counts
        let counts = zero_depth_nmap.iter().counts();
        let (most_frequent, _) = counts.into_iter()
                                        .max_by_key(|&(_, count)| count)
                                        .expect("Empty depth matrix");

        // TODO: Figure out why we want a gap in the neighborhood mapping.
        Ok((neighborhood_map.map(|&nmap_val| if nmap_val == *most_frequent { 0 } else { nmap_val }), 
            num_neighborhoods - 1))
    }
}
