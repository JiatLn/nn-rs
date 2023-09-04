mod add;
mod constructors;
mod div;
mod dot;
mod exp;
mod fmt;
mod iter;
mod minmax;
mod mul;
mod neg;
mod slice;
mod sub;
mod transpose;

use crate::{rand_standard_normal, zeros};

#[derive(Clone, PartialEq, Eq, Default)]
pub struct Matrix<T>(pub Vec<Vec<T>>);

impl<T> Matrix<T> {
    pub fn new(vec: Vec<Vec<T>>) -> Self {
        Matrix(vec)
    }
    /// (height, width)
    pub fn shape(&self) -> (usize, usize) {
        if self.0.is_empty() {
            (0, 0)
        } else {
            (self.0.len(), self.0[0].len())
        }
    }
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.0[row][col] = value;
    }
    pub fn get(&self, row: usize, col: usize) -> T
    where
        T: Copy,
    {
        self.0[row][col]
    }
}

impl Matrix<f64> {
    pub fn new_zero(height: usize, width: usize) -> Self {
        Matrix::new((0..height).map(|_| zeros(width)).collect())
    }
    pub fn sum(&self) -> f64 {
        let (h, w) = self.shape();
        (0..h).fold(0.0, |acc, i| {
            acc + (0..w).fold(0.0, |acc, j| acc + self.get(i, j))
        })
    }
    pub fn flatten(&self) -> Vec<f64> {
        self.0.clone().into_iter().flatten().collect()
    }
    pub fn new_randn(h: usize, w: usize) -> Self {
        Matrix::new(
            (0..h)
                .map(|_| (0..w).map(|_| rand_standard_normal()).collect())
                .collect(),
        )
    }
    pub fn multiple_martix_sum(&self, martix: &Matrix<f64>) -> f64 {
        let (h1, w1) = self.shape();
        let (h2, w2) = martix.shape();
        if h1 != h2 || w1 != w2 {
            panic!("shape not match");
        }
        let m = Matrix::new(
            (0..h1)
                .map(|i| (0..w1).map(|j| self.get(i, j) * martix.get(i, j)).collect())
                .collect(),
        );
        m.sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_new_zero() {
        let martix = Matrix::new_zero(3, 4);
        assert_eq!(martix.shape(), (3, 4));
    }

    #[test]
    fn test_martix_sum() {
        let matrix = Matrix::new(vec![vec![4.0, 3.0, 5.0], vec![6.0, 4.0, 3.0]]);
        let sum = matrix.sum();
        assert_eq!(sum, 25.0);
    }

    #[test]
    fn test_martix_shape() {
        let m: Matrix<f64> = Matrix::new(vec![]);
        assert_eq!(m.shape(), (0, 0));

        let m: Matrix<f64> = Matrix::new(vec![vec![]]);
        assert_eq!(m.shape(), (1, 0));

        let m = Matrix::new(vec![vec![4.0, 3.0, 5.0], vec![6.0, 4.0, 3.0]]);
        assert_eq!(m.shape(), (2, 3));
    }
}
