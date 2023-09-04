use crate::Matrix;

impl<T> Matrix<T> {
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        assert!(!self.0.is_empty());
        Matrix::new(
            (0..self.0[0].len())
                .map(|i| {
                    self.0
                        .iter()
                        .map(|inner| inner[i].clone())
                        .collect::<Vec<T>>()
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_transpose() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let m2 = m1.transpose();

        assert_eq!(m2.shape(), (3, 2));
        assert_eq!(m2.0, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    }
}
