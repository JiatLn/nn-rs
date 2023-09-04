use crate::Matrix;

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
    pub fn max(&self) -> f64 {
        self.0.iter().cloned().flatten().float_max()
    }
    pub fn min(&self) -> f64 {
        self.0.iter().cloned().flatten().float_min()
    }
    pub fn max_index(&self) -> (usize, usize) {
        let (h, w) = self.shape();
        assert!(h * w > 0);
        let mut max = f64::MIN;
        let mut index = (0, 0);
        for i in 0..h {
            for j in 0..w {
                let curr = self.get(i, j);
                if curr > max {
                    max = curr;
                    index = (i, j);
                }
            }
        }
        index
    }
    pub fn min_index(&self) -> (usize, usize) {
        let (h, w) = self.shape();
        assert!(h * w > 0);
        let mut min = f64::MAX;
        let mut index = (0, 0);
        for i in 0..h {
            for j in 0..w {
                let curr = self.get(i, j);
                if curr < min {
                    min = curr;
                    index = (i, j);
                }
            }
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martix_minmax() {
        let m = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(m.max(), 6.0);
        assert_eq!(m.min(), 1.0);
        assert_eq!(m.max_index(), (1, 2));
        assert_eq!(m.min_index(), (0, 0));
    }
}
