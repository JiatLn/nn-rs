use crate::Matrix;

impl Matrix<f64> {
    pub fn exp(&self) -> Self {
        Matrix::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|value| value.exp()).collect())
                .collect(),
        )
    }
}
