/// seathru.rs
/// Author  : Avik Ghosh
/// Purpose : Migrate individual functions from sea-thru, a library used to correct for 
///             wavelength dependent attenuation of light in underwater images.
///             Original repo: https://github.com/hainh/sea-thru
///           To be used by FishSense as library for data processing.

// Includes
extern crate nalgebra as na;
use na::DMatrix;

// Function takes in depth map (2D array) and groups them into neighborhoods of deviation
// epsilon. In other words, discretize the depth map with epsilon as an approximate precision
#[allow(dead_code)]
pub fn construct_neighborhood_map(mut depth_map: DMatrix::<f32>, epsilon: f32) -> DMatrix::<f32> {
    // Iterate through matrix and set dummy values
    for (i, mut row) in depth_map.row_iter_mut().enumerate() {
        row *= ((i as f32) + 1.0) * (10.0 + epsilon);
    }

    depth_map
}