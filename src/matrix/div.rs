use std::ops::Div;

use crate::Matrix;

impl<T> Div<T> for Matrix<T>
where
    T: Copy + std::ops::Div,
    Vec<T>: FromIterator<<T as Div>::Output>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        Self::new(
            self.0
                .iter()
                .map(|row| row.iter().map(|&v| v / rhs).collect())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_div() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let m2 = m1 / 2.0;

        assert_eq!(m2.0, vec![vec![0.5, 1.0, 1.5], vec![2.0, 2.5, 3.0]]);
    }
}
