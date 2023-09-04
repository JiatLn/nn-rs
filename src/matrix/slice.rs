use crate::Matrix;

impl<T> Matrix<T>
where
    T: Copy,
{
    pub fn slice(&self, start_row: usize, start_col: usize, size: usize) -> Self {
        let (h, w) = self.shape();
        if size > h - start_row || size > w - start_col {
            panic!("size not match")
        }
        Matrix::new(
            (start_row..start_row + size)
                .map(|i| {
                    (start_col..start_col + size)
                        .map(|j| self.get(i, j))
                        .collect()
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_slice() {
        let matrix = Matrix::new(vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![5.0, 3.0, 6.0, 4.0],
            vec![1.0, 2.0, 4.0, 3.0],
            vec![6.0, 8.0, 7.0, 9.0],
        ]);

        let slice = matrix.slice(1, 0, 2);
        assert_eq!(slice.shape(), (2, 2));
        assert_eq!(slice.0, vec![vec![5.0, 3.0], vec![1.0, 2.0]]);

        let slice = matrix.slice(1, 1, 2);
        assert_eq!(slice.shape(), (2, 2));
        assert_eq!(slice.0, vec![vec![3.0, 6.0], vec![2.0, 4.0]]);

        let slice = matrix.slice(0, 0, 4);
        assert_eq!(slice.shape(), (4, 4));
        assert_eq!(matrix.0, slice.0);
    }

    #[test]
    #[should_panic]
    fn test_matrix_slice_error() {
        let matrix = Matrix::new(vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![5.0, 3.0, 6.0, 4.0],
            vec![1.0, 2.0, 4.0, 3.0],
            vec![6.0, 8.0, 7.0, 9.0],
        ]);
        matrix.slice(3, 3, 2);
    }
}
