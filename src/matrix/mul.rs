use std::ops::Mul;

use crate::Matrix;

impl<T> Mul<T> for Matrix<T>
where
    T: Copy + std::ops::Mul,
    Vec<T>: FromIterator<<T as Mul>::Output>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Self::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v * rhs).collect())
                .collect(),
        )
    }
}

impl Mul<Matrix<f64>> for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: Matrix<f64>) -> Self::Output {
        self.dot(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_div() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let m2 = m1 * 2.0;

        assert_eq!(m2.0, vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]);
    }
}
