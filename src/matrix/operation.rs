use crate::Matrix;
use std::{
    f64,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

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
    pub fn exp(&self) -> Self {
        Matrix::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|value| value.exp()).collect())
                .collect(),
        )
    }
    pub fn max(&self) -> f64 {
        self.0.iter().cloned().flatten().float_max()
    }
    pub fn max_index(&self) -> usize {
        let mut max_idx = 0;
        let vec = self.flatten();
        let mut max = vec[0];
        for i in 1..vec.len() {
            if vec[i] > max {
                max = vec[i];
                max_idx = i;
            }
        }
        max_idx
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
        let (h2, w2) = martix.shape();

        if w1 != h2 {
            panic!("Martix shape not match!");
        }

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
}

impl Div<f64> for Matrix<f64> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            panic!("Cannot divide by zero!");
        }
        Self::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v / rhs).collect())
                .collect(),
        )
    }
}

impl Div<f64> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn div(self, rhs: f64) -> Self::Output {
        if rhs == 0.0 {
            panic!("Cannot divide by zero!");
        }
        Matrix::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v / rhs).collect())
                .collect(),
        )
    }
}

impl Mul<f64> for Matrix<f64> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v * rhs).collect())
                .collect(),
        )
    }
}

impl Mul<f64> for &Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: f64) -> Self::Output {
        Matrix::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v * rhs).collect())
                .collect(),
        )
    }
}

impl Sub<f64> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn sub(self, rhs: f64) -> Self::Output {
        Matrix::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v - rhs).collect())
                .collect(),
        )
    }
}

impl Neg for Matrix<f64> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| -v).collect())
                .collect(),
        )
    }
}

impl Add for Matrix<f64> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let (h1, w1) = self.shape();
        let (h2, w2) = other.shape();
        if w1 != w2 && h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        Matrix::new(
            (0..h1)
                .map(|i| (0..w1).map(|j| self.get(i, j) + other.get(i, j)).collect())
                .collect(),
        )
    }
}

impl Mul<&Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: &Matrix<f64>) -> Self::Output {
        self.dot(rhs)
    }
}

impl Mul<Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: Matrix<f64>) -> Self::Output {
        self.dot(&rhs)
    }
}

impl Add<&Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn add(self, other: &Matrix<f64>) -> Matrix<f64> {
        let (h1, w1) = self.shape();
        let (h2, w2) = other.shape();
        if w1 != w2 && h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        Matrix::new(
            (0..h1)
                .map(|i| (0..w1).map(|j| self.get(i, j) + other.get(i, j)).collect())
                .collect(),
        )
    }
}

impl Sub<&Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn sub(self, other: &Matrix<f64>) -> Matrix<f64> {
        let (h1, w1) = self.shape();
        let (h2, w2) = other.shape();
        if w1 != w2 && h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        Matrix::new(
            (0..h1)
                .map(|i| (0..w1).map(|j| self.get(i, j) - other.get(i, j)).collect())
                .collect(),
        )
    }
}

impl SubAssign<Matrix<f64>> for Matrix<f64> {
    fn sub_assign(&mut self, rhs: Matrix<f64>) {
        let (h1, w1) = self.shape();
        let (h2, w2) = rhs.shape();
        if w1 != w2 && h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        self.0 = (0..h1)
            .map(|i| (0..w1).map(|j| self.get(i, j) - rhs.get(i, j)).collect())
            .collect();
    }
}

impl AddAssign<Matrix<f64>> for Matrix<f64> {
    fn add_assign(&mut self, rhs: Matrix<f64>) {
        let (h1, w1) = self.shape();
        let (h2, w2) = rhs.shape();
        if w1 != w2 && h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        self.0 = (0..h1)
            .map(|i| (0..w1).map(|j| self.get(i, j) + rhs.get(i, j)).collect())
            .collect();
    }
}

impl Sub<Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn sub(self, other: Matrix<f64>) -> Matrix<f64> {
        let (h1, w1) = self.shape();
        let (h2, w2) = other.shape();
        if w1 != w2 || h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        Matrix::new(
            (0..h1)
                .map(|i| (0..w1).map(|j| self.get(i, j) - other.get(i, j)).collect())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_div() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0]; 3]);
        let m2 = m1 / 2.0;

        assert_eq!(
            m2.0,
            vec![
                vec![0.5, 1.0, 1.5],
                vec![0.5, 1.0, 1.5],
                vec![0.5, 1.0, 1.5]
            ]
        );
    }

    #[test]
    fn test_martix_dot() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0]; 3]);
        let m2 = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0]]);

        let m3 = m1 * &m2;
        assert_eq!(m3, Matrix::new(vec![vec![14.0], vec![14.0], vec![14.0]]));
    }

    #[test]
    fn test_martix_add() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0]; 3]);
        let m2 = Matrix::new(vec![vec![4.0, 1.0, 0.0]; 3]);

        let m3 = m1 + m2;
        assert_eq!(m3, Matrix::new(vec![vec![5.0, 3.0, 3.0]; 3]));
    }
}
