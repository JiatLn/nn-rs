use crate::{deriv_sigmoid, mse_loss, rand_normal, sigmoid};

#[derive(Debug)]
pub struct NeuronWork<T> {
    weights: Vec<T>,
    bias: Vec<T>,
    epochs: i32,
    learn_rate: f64,
}

impl NeuronWork<f64> {
    pub fn new(weight: i32, bias: i32) -> Self {
        let weights = (0..weight).map(|_| rand_normal()).collect::<Vec<_>>();
        let bias = (0..bias).map(|_| rand_normal()).collect::<Vec<_>>();
        NeuronWork {
            weights,
            bias,
            learn_rate: 0.1,
            epochs: 1000,
        }
    }
    pub fn feedforward(&self, x: &[f64]) -> f64 {
        let h1 = sigmoid(self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias[0]);
        let h2 = sigmoid(self.weights[2] * x[0] + self.weights[3] * x[1] + self.bias[1]);

        sigmoid(self.weights[4] * h1 + self.weights[5] * h2 + self.bias[2])
    }
    pub fn train(&mut self, data: &Vec<Vec<f64>>, all_y_trues: &Vec<f64>) -> () {
        for epoch in 0..self.epochs {
            for (x, y_true) in data.iter().zip(all_y_trues).into_iter() {
                let sum_h1 = self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias[0];
                let sum_h2 = self.weights[2] * x[0] + self.weights[3] * x[1] + self.bias[1];

                let h1 = sigmoid(sum_h1);
                let h2 = sigmoid(sum_h2);

                let sum_o1 = self.weights[4] * h1 + self.weights[5] * h2 + self.bias[2];
                let y_pred = sigmoid(sum_o1);

                // calculate partial derivatives.
                // partial L / partial w1
                let d_l_d_ypred = -2.0 * (y_true - y_pred);

                // Neron o1
                let d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1);
                let d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1);
                let d_ypred_d_b3 = deriv_sigmoid(sum_o1);

                let d_ypred_d_h1 = self.weights[4] * deriv_sigmoid(sum_o1);
                let d_ypred_d_h2 = self.weights[5] * deriv_sigmoid(sum_o1);

                // Neron h1
                let d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1);
                let d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1);

                let d_h1_d_b1 = deriv_sigmoid(sum_h1);

                // Neron h2
                let d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2);
                let d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2);

                let d_h2_d_b2 = deriv_sigmoid(sum_h2);

                // update weights and biases

                // Neruron h1
                self.weights[0] -= self.learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
                self.weights[1] -= self.learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
                self.bias[0] -= self.learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b1;
                // Neruron h2
                self.weights[2] -= self.learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
                self.weights[3] -= self.learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
                self.bias[1] -= self.learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b2;
                // Neuron o1
                self.weights[4] -= self.learn_rate * d_l_d_ypred * d_ypred_d_w5;
                self.weights[5] -= self.learn_rate * d_l_d_ypred * d_ypred_d_w6;
                self.bias[2] -= self.learn_rate * d_l_d_ypred * d_ypred_d_b3;
            }

            if epoch % 100 == 0 {
                let y_preds = data.iter().map(|x| self.feedforward(x)).collect::<Vec<_>>();
                println!(
                    "Epoch {:>4} loss: {:.4}",
                    epoch,
                    mse_loss(all_y_trues, &y_preds)
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_network_feedforward() {
        let mut network = NeuronWork::new(6, 3);
        let data = vec![
            vec![-2.0, -1.0],
            vec![25.0, 6.0],
            vec![17.0, 4.0],
            vec![-15.0, -6.0],
        ];
        let all_y_trues = vec![1.0, 0.0, 0.0, 1.0];
        network.train(&data, &all_y_trues);

        let emily = vec![-7.0, -3.0];
        let frank = vec![20.0, 2.0];

        dbg!(network.feedforward(&emily));
        dbg!(network.feedforward(&frank));
    }
}
