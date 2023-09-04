use crate::Matrix;
use std::ops::{Sub, SubAssign};

impl<T> Sub for Matrix<T>
where
    T: Copy + Sub,
    Vec<T>: FromIterator<<T as Sub>::Output>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
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

impl<T> SubAssign for Matrix<T>
where
    T: Copy + Sub,
    Vec<T>: FromIterator<<T as Sub>::Output>,
{
    fn sub_assign(&mut self, rhs: Self) {
        let (h1, w1) = self.shape();
        let (h2, w2) = rhs.shape();
        if w1 != w2 || h1 != h2 {
            panic!("m1 shape != m2 shape")
        }
        *self = Self(
            (0..h1)
                .map(|i| (0..w1).map(|j| self.get(i, j) - rhs.get(i, j)).collect())
                .collect(),
        );
    }
}

impl<T> Sub<T> for Matrix<T>
where
    T: Copy + std::ops::Sub,
    Vec<T>: FromIterator<<T as Sub>::Output>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        Self::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v - rhs).collect())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_sub() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let m2 = Matrix::new(vec![vec![3.0, 4.0, 2.0], vec![2.0, 5.0, 1.0]]);
        let m3 = m1 - m2;

        assert_eq!(m3.0, vec![vec![-2.0, -2.0, 1.0], vec![2.0, 0.0, 5.0]]);
    }

    #[test]
    fn test_martix_sub_assign() {
        let mut m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let m2 = Matrix::new(vec![vec![3.0, 4.0, 2.0], vec![2.0, 5.0, 1.0]]);

        m1 -= m2;

        assert_eq!(m1.0, vec![vec![-2.0, -2.0, 1.0], vec![2.0, 0.0, 5.0]]);
    }

    #[test]
    fn test_martix_sub_t() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let m2 = m1 - 3.0;

        assert_eq!(m2.0, vec![vec![-2.0, -1.0, 0.0], vec![1.0, 2.0, 3.0]]);
    }
}
