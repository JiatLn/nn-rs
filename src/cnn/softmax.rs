use crate::{softmax, Matrix};

pub struct SoftmaxLayer {
    pub weights: Matrix<f64>,
    pub biases: Matrix<f64>,
    pub last_input_shape: (usize, usize, usize),
    pub last_input: Matrix<f64>,
    pub last_totals: Matrix<f64>,
}

impl SoftmaxLayer {
    pub fn new(input_len: usize, nodes: usize) -> Self {
        let weights = Matrix::new_randn(input_len, nodes) / input_len as f64;
        let biases = Matrix::new_zero(1, nodes);
        SoftmaxLayer {
            weights,
            biases,
            last_input_shape: (0, 0, 0),
            last_input: Matrix::default(),
            last_totals: Matrix::default(),
        }
    }

    pub fn forward(&mut self, input: &Vec<Matrix<f64>>) -> Matrix<f64> {
        let (h, w) = input[0].shape();

        self.last_input_shape = (input.len(), h, w);

        let input = Matrix::new(vec![input.iter().map(|m| m.flatten()).flatten().collect()]);

        self.last_input = input.clone();

        let totals = input * self.weights.clone() + self.biases.clone();

        self.last_totals = totals.clone();

        softmax(&totals)
    }

    /// lr means learn rate
    pub fn backprop(&mut self, d_l_d_out: &Matrix<f64>, lr: f64) -> Vec<Matrix<f64>> {
        let d_l_d_out = d_l_d_out.flatten();

        let mut d_l_d_inputs = Matrix::default();

        for (i, &gradient) in d_l_d_out.iter().enumerate() {
            if gradient == 0.0 {
                continue;
            }
            // e^totals
            let t_exp = self.last_totals.exp();

            // Sum of all e^totals
            let sum = t_exp.sum();

            // Gradients of out[i] against totals
            let t_exp_i = t_exp.get(0, i);
            let mut d_out_d_t = t_exp * -t_exp_i / sum.powf(2.0);
            let new_value = t_exp_i * (sum - t_exp_i) / sum.powf(2.0);
            d_out_d_t.set(0, i, new_value);

            // Gradients of totals against weights/biases/input
            let d_t_d_w = &self.last_input;
            let d_t_d_b = 1.0;
            let d_t_d_inputs = &self.weights;

            // Gradients of loss against totals
            let d_l_d_t = d_out_d_t * gradient;

            // Gradients of loss against weights/biases/input
            let d_l_d_w = d_t_d_w.transpose().dot(&d_l_d_t);
            let d_l_d_b = d_l_d_t.clone() * d_t_d_b;
            d_l_d_inputs = d_t_d_inputs.dot(&d_l_d_t.transpose());

            // Update weights / biases
            self.weights -= d_l_d_w * lr;
            self.biases -= d_l_d_b * lr;
        }
        Matrix::from_shape_vec(&d_l_d_inputs.flatten(), self.last_input_shape)
    }
}
