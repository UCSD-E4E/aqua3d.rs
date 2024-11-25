mod seathru;
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    // Includes
    use seathru::{RgbdData, Seathru};

    #[allow(unused)]
    use std::collections::VecDeque;
    use anyhow::{Context, Result};

    #[allow(unused)]
    use num_traits::ToPrimitive;

    #[allow(unused)]
    use ndarray::{array, Array1, Array2};
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn construct_neighborhood_map_test() -> Result<()> {
        let depth_matrix = array![
            [0.001,  0.003,  7.24,   8.11,   9.87],  // Top row (random near 0 and top right)
            [0.004,  0.002,  5.18,   6.92,   4.23],  // Second row
            [10.01,  11.45,  0.23,   13.76,  14.54], // Third row (random middle)
            [20.12,  19.67,  17.35,  1.004,  1.002], // Fourth row (bottom right near 1)
            [21.11,  23.05,  22.98,  1.001,  1.003]  // Fifth row
        ];

        let img_data =  Array2::<u8>::zeros((3, 3));

        #[allow(unused)]
        let image = RgbdData {
            img : &img_data,
            depth_map : &depth_matrix
        };

        let _res = image.run_pipeline(false)?;
        let (nmap, _num_neighborhoods) = image.construct_neighborhood_map(0.1, false)
                                                                                        .context("Failed to set up neighborhood map")?;

        dbg!(&nmap);
        dbg!(&depth_matrix);
        // let (nmap, num_neighborhoods) = seathru::construct_neighborhood_map(&depth_matrix, 0.1, false)
                                                                                    // .context("Failed to set up neighborhood map")?;

        Ok(())
    }   

}
