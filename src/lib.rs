mod cnn;
mod matrix;
mod nn;
mod utils;

pub use cnn::{ConvLayer, MaxPoolLayer};
pub use matrix::Matrix;
pub use nn::{Neuron, NeuronWork};
pub use utils::*;

pub type Vec2<T> = Vec<Vec<T>>;
pub type Vec3<T> = Vec<Matrix<T>>;
