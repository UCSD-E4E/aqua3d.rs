/// seathru.rs
/// Author  : Avik Ghosh
/// Purpose : Migrate individual functions from sea-thru, a library used to correct for 
///             wavelength dependent attenuation of light in underwater images.
///             Original repo: https://github.com/hainh/sea-thru
///           To be used by FishSense as library for data processing.

// Includes
extern crate nalgebra as na;
use na::DMatrix;

use std::collections::VecDeque;
use num_traits::ToPrimitive;
use rand::Rng;

// Allow for addition of indices
macro_rules! add_tuples {
    ($t1:expr, $t2:expr) => {
        ($t1.0 + $t2.0, $t1.1 + $t2.1)
    };
}

// Extend DMatrix
trait NumPyMethods {
    fn np_where(&self, val: i32) -> Vec<(usize, usize)>;
    fn np_unique_cnts(&self) -> Vec<(i32, usize)>;
    fn is_within_bounds(&self, index : (i32, i32)) -> bool;
}

impl NumPyMethods for DMatrix<i32> 
// where 
    // T: RealField + ToPrimitive,
{
    fn np_where(&self, val: i32) -> Vec::<(usize, usize)> {
        let mut inds = Vec::<(usize, usize)>::new();
        let mut ind = 0;

        let _ = DMatrix::from_fn(self.nrows(), self.ncols(), |i, j| {
            let mat_val = self[(i, j)];
            if mat_val.to_i32() == Some(val) { inds.insert(ind, (i, j)); ind += 1; }
        });
        
        inds
    }

    fn np_unique_cnts(&self) -> Vec<(i32, usize)> {
        let mut vals = Vec::<i32>::new();
        let mut counts = Vec::<usize>::new();

        let _ = DMatrix::from_fn(self.nrows(), self.ncols(), |i, j| {
            let mat_val = self[(i, j)];
            if vals.contains(&mat_val) { 
                counts[vals.iter().position(|&val| val == mat_val).unwrap() as usize] += 1; 
            } else {
                vals.push(mat_val);
                counts.push(1);
            }
        });

        let unique = vals.iter().zip(counts.iter())
                .map(|(&x, &y)| (x, y))
                .collect();
        unique    
    } 

    fn is_within_bounds(&self, index : (i32, i32)) -> bool {
        if index.0 < 0 || index.1 < 0 {
            false
        } else if index.0 >= (self.nrows() as i32) || index.1 >= (self.ncols() as i32) {
            false
        } else {
            true
        }
    }
}

// Function takes in depth map (2D array) and groups them into neighborhoods of deviation
// epsilon. In other words, discretize the depth map with epsilon as an approximate precision
// Each 'neighborhood' is computed to be in the same epsilon-neighborhood.
#[allow(dead_code)]
pub fn construct_neighborhood_map(depth_map: &DMatrix::<f32>, epsilon: f32) -> (DMatrix::<i32>, i32) {
    // Copying original functionality
    let scaled_eps = (depth_map.max() - depth_map.min()) * epsilon;
    let mut neighborhood_map = DMatrix::<i32>::zeros(depth_map.shape().0, depth_map.shape().1);
    let mut num_neighborhoods = 1;

    // Can convert into .min() == 0 for memory efficiency
    while !neighborhood_map.np_where(0).is_empty() {                         
        let zero_map = neighborhood_map.np_where(0);
        let start_index = rand::thread_rng().gen_range(0..zero_map.len());
        let (start_x, start_y) = zero_map[start_index];

        let mut q = VecDeque::<(usize, usize)>::new();
        q.push_back((start_x, start_y));

        let mut x_ind : i32;
        let mut y_ind : i32;
        while q.len() != 0 {
            let first_val = q.pop_front().unwrap();
            (x_ind, y_ind) = (first_val.0 as i32, first_val.1 as i32);

            if (depth_map[(x_ind as usize, y_ind as usize)] - depth_map[(start_x, start_y)]).abs() <= scaled_eps {
                neighborhood_map[(x_ind as usize, y_ind as usize)] = num_neighborhoods;

                // Add in neighboring points. x_ind and y_ind are already positive or 0, so no need to check that.
                // List of Possible Deltas from (x_ind, y_ind) to have the 4 surrounding (non-diagonal) points.
                let deltas = vec![(-1, 0), (1, 0), (0, -1), (0, 1)];
                for delta in deltas {
                    let (del_x_ind, del_y_ind) = add_tuples!((x_ind, y_ind), delta);
                    if (neighborhood_map.is_within_bounds((del_x_ind, del_y_ind))) &&       // Needs to short circuit
                        neighborhood_map[(del_x_ind as usize, del_y_ind as usize)] == 0 {
                            q.push_back((del_x_ind as usize, del_y_ind as usize));
                    }
                }
            }
        }
        num_neighborhoods += 1; // Move onto the next neighborhood
    }

    // TODO: Why are we doing equality on floating point?? is depth integer then? Currently using 
    let practically_zero = 0.5;
    let zero_depth = depth_map.zip_map(&depth_map, |a, _b| a.abs() < practically_zero);
    let nmap_on_zero_depth = neighborhood_map.zip_map
    (
        &zero_depth, 
        |nmap_value, is_zero| {
            if is_zero { nmap_value } else { -1 }
        }
    );

    // let mut vals_w_counts = nmap_on_zero_depth.np_unique_cnts();
    let mut vals_w_counts = nmap_on_zero_depth.np_unique_cnts();
    vals_w_counts.retain(|&(nmap_val, _)| nmap_val != -1);
    vals_w_counts.sort_by_key(|tup| tup.1);                      // Sort in ascending order on count
    vals_w_counts.reverse();                                                    // Reverse so max count at beginning

    if vals_w_counts.len() > 0 { 
        // Reset largest nmap count to be 0
        neighborhood_map = neighborhood_map.zip_map(&neighborhood_map, |nmap_val, _b| if nmap_val == vals_w_counts[0].0 { 0 } else { nmap_val } );
    }
    
    (neighborhood_map, num_neighborhoods - 1)
}