use rand_distr::{Distribution, Normal};

pub fn rand_normal() -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.sample(&mut rand::thread_rng())
}
