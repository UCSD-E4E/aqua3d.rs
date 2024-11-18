mod seathru;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_construct_neighborhood_map() {
        // let depth_matrix = DMatrix::<f32>::from_data(data);
        let depth_matrix = DMatrix::<f32>::from_row_slice(5, 5, &[
            0.001,  0.003,  7.24,   8.11,   9.87,  // Top row (random near 0 and top right)
            0.004,  0.002,  5.18,   6.92,   4.23,  // Second row
            10.01,  11.45,  0.23,   13.76,  14.54, // Third row (random middle)
            20.12,  19.67,  17.35,  1.004,  1.002, // Fourth row (bottom right near 1)
            21.11,  23.05,  22.98,  1.001,  1.003  // Fifth row
        ]);
        let nmap : DMatrix::<i32>;
        let num_neighborhoods : i32;

        (nmap, num_neighborhoods) = seathru::construct_neighborhood_map(&depth_matrix, 0.3);

        println!("Depth Matrix: ");
        for (_, row) in depth_matrix.row_iter().enumerate() {
            print!("[");
            for (_, column) in row.into_iter().enumerate() {
                print!("{}\t", column);
            }
            println!("]");
        }
        println!();
        println!("Neighborhood Map: ");

        for (_, row) in nmap.row_iter().enumerate() {
            print!("[");
            for (_, column) in row.into_iter().enumerate() {
                print!("{}\t", column);
            }
            println!("]");
        }
        // dbg!(depth_matrix);
        // dbg!(nmap);
        println!("There were {} neighborhoods!", num_neighborhoods)

    }
}
