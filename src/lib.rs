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
            1.0001, 0.0, 0.0, 1.0004, 1.0005,  // First row (values close to each other)
            2.0,    0.0,    0.0,    5.0,    6.0,    // Second row
            7.0,    8.0,    9.0,    0.0,   11.0,   // Third row
            12.0,   13.0,   0.0,   15.0,   16.0,   // Fourth row
            17.0,   18.0,   19.0,   20.0,   21.0    // Fifth row
        ]);
        let nmap : DMatrix::<i32>;
        let num_neighborhoods : i32;

        (nmap, num_neighborhoods) = seathru::construct_neighborhood_map(&depth_matrix, 0.03);

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
