use crate::rand_standard_normal;

pub fn zeros(len: usize) -> Vec<f64> {
    vec![0.0; len]
}

pub fn randn_martix(height: usize, width: usize) -> Vec<Vec<f64>> {
    (0..height)
        .map(|_| (0..width).map(|_| rand_standard_normal()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let vec = zeros(4);
        assert_eq!(vec, vec![0.0, 0.0, 0.0, 0.0])
    }

    #[test]
    fn test_martix_with_randn() {
        let vec = randn_martix(3, 4);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0].len(), 4);
    }
}
