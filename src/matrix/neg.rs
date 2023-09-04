use crate::Matrix;
use std::ops::Neg;

impl<T> Neg for Matrix<T>
where
    Vec<T>: FromIterator<<T as Neg>::Output>,
    T: Neg + Copy,
{
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_neg() {
        let m1 = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let m2 = -m1;

        assert_eq!(m2.0, vec![vec![-1.0, -2.0, -3.0], vec![-4.0, -5.0, -6.0]]);
    }
}
