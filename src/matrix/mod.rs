mod constructors;
mod fmt;
mod iter;
mod operation;
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
        assert!(!self.0.is_empty());
        (self.0.len(), self.0[0].len())
    }
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.0[row][col] = value;
    }
}

impl Matrix<f64> {
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.0[row][col]
    }
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
    pub fn new_randn(height: usize, width: usize) -> Self {
        Matrix::new(
            (0..height)
                .map(|_| (0..width).map(|_| rand_standard_normal()).collect())
                .collect(),
        )
    }
    pub fn slice(&self, start_row: usize, start_col: usize, size: usize) -> Self {
        let mut vec = Vec::with_capacity(size);
        for i in start_row..start_row + size {
            let mut row = Vec::with_capacity(size);
            for j in start_col..start_col + size {
                row.push(self.0[i][j]);
            }
            vec.push(row);
        }
        Matrix::new(vec)
    }
    pub fn multiple_martix_sum(&self, martix: &Matrix<f64>) -> f64 {
        let (w, h) = self.shape();
        let mut result = 0.0;
        for i in 0..h {
            for j in 0..w {
                result += self.get(i, j) * martix.get(i, j);
            }
        }
        result
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
    fn test_martix_slice() {
        let matrix = Matrix::new(vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![5.0, 3.0, 6.0, 4.0],
            vec![1.0, 2.0, 4.0, 3.0],
            vec![6.0, 8.0, 7.0, 9.0],
        ]);
        let slice = matrix.slice(1, 0, 2);
        assert_eq!(slice.shape(), (2, 2));
    }

    #[test]
    fn test_martix_sum() {
        let matrix = Matrix::new(vec![vec![4.0, 3.0, 5.0], vec![6.0, 4.0, 3.0]]);
        let sum = matrix.sum();
        assert_eq!(sum, 25.0);
    }
}
