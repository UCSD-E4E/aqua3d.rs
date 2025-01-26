use ndarray::{s, Array2, Array3};
use ndarray_stats::QuantileExt;

use crate::errors::Aqua3dError;

fn update_a_prime(a_values: &Array3<f64>, neighborhood_map: &Array2<u32>) -> Result<Array3<f64>, Aqua3dError> {
    let max_class = neighborhood_map.max()?;

    let mut a_prime = Array3::<f64>::zeros(a_values.dim());
    for i in 1..(max_class + 1) {
        let selected_cells = neighborhood_map.mapv(|f| if f == i { 1f64 } else { 0f64 });
        let selected_count = selected_cells.sum();
        if selected_count == 0f64 {
            continue;
        }

        a_prime.slice_mut(s![.., .., 0]).assign(&(&selected_cells * (&a_values.slice(s![.., .., 0]) * &selected_cells).sum() / selected_count));
        a_prime.slice_mut(s![.., .., 1]).assign(&(&selected_cells * (&a_values.slice(s![.., .., 1]) * &selected_cells).sum() / selected_count));
        a_prime.slice_mut(s![.., .., 2]).assign(&(&selected_cells * (&a_values.slice(s![.., .., 2]) * &selected_cells).sum() / selected_count));
    }

    return Ok(a_prime);
}

fn update_a_values(direct_signal: &Array3<f64>, a_prime: &Array3<f64>, p: f64) -> Array3<f64> {
    return direct_signal * p + a_prime * (1f64 - p);
}

pub fn compute_local_space_average(direct_signal: &Array3<f64>, neighborhood_map: &Array2<u32>, convergence_threshold: f64, p: f64) -> Result<Array3<f64>, Aqua3dError> {
    let (height, width, channels) = direct_signal.dim();

    let mut a_values = Array3::<f64>::zeros((height, width, channels));
    let mut prev_a_values = Array3::<f64>::ones((height, width, channels));

    while (&a_values - &prev_a_values).abs().iter().all(|f| f > &convergence_threshold) {
        println!("{}", (&a_values - &prev_a_values).abs().max()?);

        let a_prime = update_a_prime(&a_values, neighborhood_map)?;
        prev_a_values = a_values;

        a_values = update_a_values(direct_signal, &a_prime, p);
    }

    return Ok(a_values);
}