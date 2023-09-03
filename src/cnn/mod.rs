mod cnn;
mod conv;
mod maxpool;
mod softmax;

pub use cnn::{CNNConfig, CNNOutput, CNN};
pub use conv::ConvLayer;
pub use maxpool::MaxPoolLayer;
pub use softmax::SoftmaxLayer;
