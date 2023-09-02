use crate::{rand_standard_normal, Vec2};

pub fn zeros(size: usize) -> Vec<f64> {
    vec![0.0; size]
}

pub fn randn_martix(h: usize, w: usize) -> Vec2<f64> {
    (0..h)
        .map(|_| (0..w).map(|_| rand_standard_normal()).collect())
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
