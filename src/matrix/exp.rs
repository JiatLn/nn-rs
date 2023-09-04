use crate::Matrix;

impl Matrix<f64> {
    pub fn exp(&self) -> Matrix<f64> {
        Matrix::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|value| value.exp()).collect())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_exp() {
        let m1 = Matrix::new(vec![vec![0.89062689, 0.87994127, 0.94568461]]);
        let m2 = m1.exp();

        assert_eq!(
            m2.0,
            vec![vec![
                2.4366566883090806,
                2.410758118435224,
                2.5745753554201025
            ]]
        );
    }
}
