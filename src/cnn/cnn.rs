use crate::{ConvLayer, Matrix, MaxPoolLayer, SoftmaxLayer};

pub struct CNN {
    conv: ConvLayer,
    maxpool: MaxPoolLayer,
    softmax: SoftmaxLayer,
}

pub struct CNNConfig {
    pub filter_size: usize,
    pub filter_num: usize,
    pub pool_size: usize,
    pub input_len: usize,
    pub nodes: usize,
}

#[allow(dead_code)]
pub struct CNNOutput {
    out: Matrix<f64>,
    loss: f64,
    acc: bool,
}

impl CNN {
    pub fn new(config: CNNConfig) -> Self {
        let conv = ConvLayer::new(config.filter_size, config.filter_num);
        let maxpool = MaxPoolLayer::new(config.pool_size);
        let softmax = SoftmaxLayer::new(config.input_len, config.nodes);
        CNN {
            conv,
            maxpool,
            softmax,
        }
    }
    fn forward(&self, image: &Matrix<f64>, label: usize) -> CNNOutput {
        let out = self.conv.forward(image);
        let out = self.maxpool.forward(&out);
        let out = self.softmax.forward(&out);

        let loss = -out.0[0][label].ln();
        let acc = out.max_index() == label;

        CNNOutput { out, loss, acc }
    }
    pub fn run(self, test_images: &Vec<Matrix<f64>>, test_labels: &Vec<usize>) -> () {
        let mut loss = 0.0;
        let mut num_correct = 0;

        for (idx, (img, &label)) in test_images.iter().zip(test_labels).enumerate() {
            let output = self.forward(img, label);

            loss += output.loss;
            num_correct += if output.acc { 1 } else { 0 };

            if idx % 100 == 99 {
                println!(
                    "[Step {:>4}] Past 100 steps: Average Loss {:.4} | Accuracy: {}%",
                    idx + 1,
                    loss / 100.0,
                    num_correct
                );
                loss = 0.0;
                num_correct = 0;
            }
        }
    }
}
