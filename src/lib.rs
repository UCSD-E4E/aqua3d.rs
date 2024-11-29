mod seathru;
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    // use itertools::Itertools;
    // use ndarray::Slice;
    // use num_traits::zero;
    // Includes
    #[allow(unused)]
    use seathru::{RgbdData, Seathru};

    #[allow(unused)]
    use anyhow::{Context, Result};
    #[allow(unused)]
    use std::collections::VecDeque;

    #[allow(unused)]
    use num_traits::ToPrimitive;

    use super::*;
    #[allow(unused)]
    use ndarray::{array, s, Array1, Array2, Array3, Axis};

    // use rand::{distributions::Uniform, Rng};

    // #[test]
    // fn it_works() {
    //     let result = add(2, 2);
    //     assert_eq!(result, 4);
    // }

    #[test]
    fn construct_neighborhood_map_test() -> Result<()> {
        #[allow(unused)]
        let depth_matrix = array![
            [0.001, 0.003, 7.24, 8.11, 9.87], // Top row (random near 0 and top right)
            [0.004, 0.002, 5.18, 6.92, 4.23], // Second row
            [10.01, 11.45, 0.23, 13.76, 14.54], // Third row (random middle)
            [20.12, 19.67, 17.35, 1.004, 1.002], // Fourth row (bottom right near 1)
            [21.11, 23.05, 22.98, 1.001, 1.003]  // Fifth row
        ];

        // let mut rng = rand::thread_rng();
        // let range = Uniform::new(0, 20);
        let l = depth_matrix.nrows();
        let w = depth_matrix.ncols();

        let depth = depth_matrix.flatten().to_vec();
        let depth_vec = [&depth[..], &depth[..], &depth[..]].concat();

        let rand_data: Vec<f64> = (0..(l * w * 3)).map(|v| depth_vec[v]).collect();
        let _img_data: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 3]>> =
            dbg!(Array3::<f64>::from_shape_vec((l, w, 3), rand_data).unwrap());
        dbg!(&_img_data.slice(s![0, 0, ..]));

        // println!("{:?}", img_data.select(Axis(0), &[1]).select(Axis(1), &[1]).sum());
        // let x = Array2::from_elem((l, w), 1.0);
        // let _z : Array2<f64> = Array2::from_shape_vec((l, w),
        // x.indexed_iter().map(|((i, j), _)| img_data.select(Axis(0), &[i]).select(Axis(1), &[j]).sum()).collect())?;

        // to owned??
        // let test_ara : Vec<Array1<f64>> = locs.map(|(index, _)| img_data.select(Axis(0), &[index.0])
        //                                                                                 .select(Axis(1), &[index.1])
        //                                                                                 .flatten()
        //                                                                                 .clone().to_owned()
        //                                             ).collect();
        // dbg!(&test_ara);
        // let sum_arr = Array2::from_elem((l, w), 1.0).indexed_iter().map(|((i, j), _val)| img_data.select(Axis(0), &[i]).select(Axis(1), &[j])).collect();
        // dbg!(img_data);
        // dbg!(summed_arr);
        // for i in rgb_slices {
        // println!("{:?}", i);
        // println!("Testing!")
        // }

        // #[allow(unused)]
        // let image = RgbdData {
        //     img : &img_data,
        //     depth_map : &depth_matrix
        // };

        // let _res = image.run_pipeline(false)?;
        // let (nmap, _num_neighborhoods) = image.construct_neighborhood_map(0.1, false)
        //                                                                                 .context("Failed to set up neighborhood map")?;

        // dbg!(&nmap);
        // dbg!(&depth_matrix);
        // let (nmap, num_neighborhoods) = seathru::construct_neighborhood_map(&depth_matrix, 0.1, false)
        // .context("Failed to set up neighborhood map")?;

        Ok(())
    }
}
