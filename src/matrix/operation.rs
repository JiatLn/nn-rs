use crate::Matrix;
use std::f64;

trait FloatIterExt {
    fn float_min(&mut self) -> f64;
    fn float_max(&mut self) -> f64;
}

impl<T> FloatIterExt for T
where
    T: Iterator<Item = f64>,
{
    fn float_max(&mut self) -> f64 {
        self.fold(f64::NAN, f64::max)
    }

    fn float_min(&mut self) -> f64 {
        self.fold(f64::NAN, f64::min)
    }
}

impl Matrix<f64> {
    pub fn drivide(mut self, divider: f64) -> Self {
        self.0 = self
            .0
            .iter()
            .map(|row| row.iter().map(|value| value / divider).collect())
            .collect();
        self
    }
    pub fn exp(mut self) -> Self {
        self.0 = self
            .0
            .iter()
            .map(|row| row.iter().map(|value| value.exp()).collect())
            .collect();
        self
    }
    pub fn max(&self) -> f64 {
        self.0.iter().cloned().flatten().float_max()
    }
    pub fn min(&self) -> f64 {
        self.0.iter().cloned().flatten().float_min()
    }
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
    pub fn dot(&self, martix: &Matrix<f64>) -> Self {
        let (h1, w1) = self.shape();
        let (_h2, w2) = martix.shape();

        Matrix::new(
            (0..h1)
                .map(|i| {
                    (0..w2).fold(Vec::with_capacity(w2), |mut acc, j| {
                        acc.push(
                            (0..w1)
                                .map(|k| self.0[i][k] * martix.0[k][j])
                                .fold(0.0, |acc, x| acc + x),
                        );
                        acc
                    })
                })
                .collect(),
        )
    }
    pub fn add(&self, martix: &Matrix<f64>) -> Self {
        let (h1, w1) = self.shape();
        let (h2, w2) = martix.shape();
        if w1 != w2 && h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        Matrix::new(
            (0..h1)
                .map(|i| (0..w1).map(|j| self.get(i, j) + martix.get(i, j)).collect())
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
        assert_eq!(m3, Matrix::new(vec![vec![14.0], vec![14.0], vec![14.0]]));
    }

    #[test]
    fn test_martix_add() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0]; 3]);
        let m2 = Matrix::new(vec![vec![4.0, 1.0, 0.0]; 3]);

        let m3 = m1.add(&m2);
        assert_eq!(m3, Matrix::new(vec![vec![5.0, 3.0, 3.0]; 3]));
    }
}
