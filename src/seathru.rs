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

// Extend DMatrix
trait NumPyMethods {
    fn np_where(&self, val: i32) -> Vec<(usize, usize)>;
    fn np_unique_cnts(&self) -> Vec<(i32, usize)>;
}

impl NumPyMethods for DMatrix<i32> 
// where 
    // T: RealField + ToPrimitive,
{
    fn np_where(&self, val: i32) -> Vec::<(usize, usize)> {
        let mut inds = Vec::<(usize, usize)>::new();
        let mut ind = 0;
        
        for (i, row) in self.row_iter().enumerate() {
            for (j, column) in row.into_iter().enumerate() {
                if column.to_i32() == Some(val) {
                    inds.insert(ind, (i as usize, j as usize));
                    ind += 1;
                }
            }
        } 
        
        // inds now contains all values that would be cast into val from f32 to i32
        inds
    }

    fn np_unique_cnts(&self) -> Vec<(i32, usize)> {
        let mut vals = Vec::<i32>::new();
        let mut counts = Vec::<usize>::new();

        let mut temp_ind: i32;
        for (_, row) in self.row_iter().enumerate() {
            print!("[");
            for (_, column) in row.into_iter().enumerate() {
                print!("{}\t", column);
                if !vals.contains(column) {
                    vals.push(*column);
                    counts.push(1);
                } else {
                    temp_ind = match vals.iter().position(|&val| val == *column) {
                        Some(index) => index as i32,
                        None => -1 ,            
                    };

                    // println!("Incrementing {} at index {} to {}", *column, temp_ind, counts[temp_ind as usize]);

                    if temp_ind == -1 {
                        // Should not run into this as we already saw vals.contains(column) should be true
                        println!("{} not found. Failed to find unique counts for matrix.", column);
                    } else {
                        counts[temp_ind as usize] += 1;
                    }

                }
            }
            println!("]");
        }

        let unique = vals.iter().zip(counts.iter())
                .map(|(&x, &y)| (x, y))
                .collect();
        unique    

    } 
}

// Function takes in depth map (2D array) and groups them into neighborhoods of deviation
// epsilon. In other words, discretize the depth map with epsilon as an approximate precision
#[allow(dead_code)]
pub fn construct_neighborhood_map(depth_map: &DMatrix::<f32>, epsilon: f32) -> (DMatrix::<i32>, i32) {
    // Copying original functionality
    let scaled_eps = (depth_map.max() - depth_map.min()) * epsilon;
    let mut neighborhood_map = DMatrix::<i32>::zeros(depth_map.shape().0, depth_map.shape().1);
    let mut num_neighborhoods = 1;

    // Quicker alternative to neighborhood map .any(neighborhood_map == 0), all neighborhood_map vals > 0
    while neighborhood_map.min() == 0 {   
        let zero_map = neighborhood_map.np_where(0);
        let start_index = rand::thread_rng().gen_range(0..zero_map.len());
        let (start_x, start_y) = zero_map[start_index];

        let mut q = VecDeque::<(usize, usize)>::new();
        q.push_back((start_x, start_y));

        let mut x_ind : usize;
        let mut y_ind : usize;
        while q.len() != 0 {
            (x_ind, y_ind) = q.pop_front().unwrap();
            if (depth_map[(x_ind, y_ind)] - depth_map[(start_x, start_y)]).abs() <= scaled_eps {
                neighborhood_map[(x_ind, y_ind)] = num_neighborhoods;

                // Add in neighboring points. x_ind and y_ind are already positive or 0, so no need to check that.
                // Check (x+1, y)
                if (x_ind < depth_map.shape().0 - 1) && 
                    neighborhood_map[(x_ind + 1, y_ind)] == 0 {
                    q.push_back((x_ind + 1, y_ind));
                }

                // Check (x-1, y)
                if (1 <= x_ind && x_ind < depth_map.shape().0) && 
                    neighborhood_map[(x_ind - 1, y_ind)] == 0 {
                    q.push_back((x_ind - 1, y_ind));
                }

                // Check (x, y+1)
                if (y_ind < depth_map.shape().1 - 1) && 
                    neighborhood_map[(x_ind, y_ind + 1)] == 0 {
                    q.push_back((x_ind, y_ind + 1));
                }

                // Check (x, y-1)
                if (1 <= y_ind && y_ind < depth_map.shape().0) && 
                    neighborhood_map[(x_ind, y_ind - 1)] == 0 {
                    q.push_back((x_ind, y_ind - 1));
                }

            }
        }

        // nth neighborhood is completed (grouped all possible depths together in a group of size epsilon)
        // Move onto the next group
        num_neighborhoods += 1;
    }

    // TODO: Why are we doing equality on floating point?? is depth integer then? Currently using 
    let practically_zero = 1e-5;
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
    vals_w_counts.sort_by_key(|tup| tup.1);                     // Sort in ascending order on count
    vals_w_counts.reverse();                                                 // Reverse so max count at beginning

    print!("[");
    for (x, y) in &vals_w_counts {
        println!("({}, {}), ", x, y);
    }
    print!("]");

    if vals_w_counts.len() > 0 {
        // Reset largest nmap count to be 0. We use vals_w_counts[1] 
        let mut proper_ind : usize = 0;
        for i in 0..vals_w_counts.len() {
            if vals_w_counts[i].0 == -1 {
                proper_ind += 1;
            }
        }
        neighborhood_map = neighborhood_map.zip_map(&neighborhood_map, |nmap_val, _b| if nmap_val == vals_w_counts[proper_ind].0 { 0 } else { nmap_val } );
    }
    
    (neighborhood_map, num_neighborhoods - 1)
}