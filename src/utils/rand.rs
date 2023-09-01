use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal};

pub fn rand_normal() -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.sample(&mut thread_rng())
}

pub fn rand_standard_normal() -> f64 {
    thread_rng().sample(StandardNormal)
}
