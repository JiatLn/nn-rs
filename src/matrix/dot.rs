use crate::Matrix;

impl Matrix<f64> {
    /// <https://www.w3.org/TR/css-color-4/multiply-matrices.js>
    ///
    /// a is m x n. b is n x p. product is m x p.
    ///
    /// a:
    /// ```bash
    /// | 1, 2, 3 |
    /// | 4, 5, 6 |
    /// | 7, 8, 9 |
    /// ```
    /// b:
    /// ```bash
    /// | 1 |
    /// | 2 |
    /// | 3 |
    /// ```
    /// product:
    /// ```bash
    /// | 14 |
    /// | 32 |
    /// | 50 |
    /// ```
    pub fn dot(&self, other: &Matrix<f64>) -> Matrix<f64> {
        let (h1, w1) = self.shape();
        let (h2, w2) = other.shape();

        if w1 != h2 {
            panic!("Martix shape not match!");
        }

        Matrix::new(
            (0..h1)
                .map(|i| {
                    (0..w2).fold(Vec::with_capacity(w2), |mut acc, j| {
                        acc.push(
                            (0..w1)
                                .map(|k| self.get(i, k) * other.get(k, j))
                                .fold(0.0, |acc, x| acc + x),
                        );
                        acc
                    })
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_dot() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0]; 3]);
        let m2 = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]);

        let m3 = m1.dot(&m2);
        assert_eq!(m3.0, vec![vec![14.0], vec![14.0], vec![14.0]]);

        let m1 = Matrix::new(vec![vec![1.0, 0.0, 2.0], vec![-1.0, 3.0, 1.0]]);
        let m2 = Matrix::new(vec![vec![3.0, 1.0], vec![2.0, 1.0], vec![1.0, 0.0]]);

        let m3 = m1.dot(&m2);
        assert_eq!(m3.0, vec![vec![5.0, 1.0], vec![4.0, 2.0]]);
    }
}
