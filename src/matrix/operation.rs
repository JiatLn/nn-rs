use crate::Matrix;
use std::f64;

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
    pub fn drivide(mut self, divider: f64) -> Self {
        self.0 = self
            .0
            .iter()
            .map(|row| row.iter().map(|value| value / divider).collect())
            .collect();
        self
    }
    pub fn max(&self) -> f64 {
        self.0.iter().cloned().flatten().float_max()
    }
    pub fn min(&self) -> f64 {
        self.0.iter().cloned().flatten().float_min()
    }
}
