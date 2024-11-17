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
        let mut depth_matrix = DMatrix::<f32>::from_element(3, 3, 1.0);
        depth_matrix = seathru::construct_neighborhood_map(depth_matrix, 0.03);

        for (_, row) in depth_matrix.row_iter().enumerate() {
            print!("[");
            for (_, column) in row.into_iter().enumerate() {
                print!("{} ", column);
            }
            println!("]");
        }
    }
}
