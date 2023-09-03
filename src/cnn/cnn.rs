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
    fn forward(&mut self, image: &Matrix<f64>, label: usize) -> CNNOutput {
        let input = image / 255.0 - 0.5;
        let out = self.conv.forward(&input);
        let out = self.maxpool.forward(&out);
        let out = self.softmax.forward(&out);

        let loss = -out.get(0, label).ln();
        let acc = out.max_index() == label;

        CNNOutput { out, loss, acc }
    }
    fn train(&mut self, image: &Matrix<f64>, label: usize, lr: f64) -> CNNOutput {
        // Forward
        let output = self.forward(image, label);

        // Calculate initial gradient
        let mut gradient = Matrix::new_zero(1, 10);
        gradient.set(0, label, -1.0 / output.out.get(0, label));

        // Backprop
        let gradient = self.softmax.backprop(&gradient, lr);
        let gradient = self.maxpool.backprop(&gradient);
        self.conv.backprop(&gradient, lr);

        output
    }
    pub fn run(&mut self, train_images: &[Matrix<f64>], train_labels: &[usize]) -> () {
        assert_eq!(train_images.len(), train_labels.len());

        let mut loss = 0.0;
        let mut correct_num = 0;

        for (idx, (image, &label)) in train_images.iter().zip(train_labels).enumerate() {
            if idx % 100 == 99 {
                println!(
                    "[Step {:>4}] Past 100 steps: Average Loss {:.4} | Accuracy: {}%",
                    idx + 1,
                    loss / 100.0,
                    correct_num
                );
                loss = 0.0;
                correct_num = 0;
            }

            let output = self.train(image, label, 0.005);
            loss += output.loss;
            correct_num += if output.acc { 1 } else { 0 };
        }
    }
}
