use crate::{randn_martix, Matrix};

pub struct SoftmaxLayer {
    pub weights: Matrix<f64>,
    pub biases: Matrix<f64>,
}

impl SoftmaxLayer {
    pub fn new(input_len: usize, nodes: usize) -> Self {
        let weights = Matrix::new(randn_martix(input_len, nodes)).drivide(input_len as f64);
        let biases = Matrix::new_zero(nodes, 1);
        SoftmaxLayer { weights, biases }
    }

    pub fn forward(&self, input: &Vec<Matrix<f64>>) -> Matrix<f64> {
        let input = input
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let totals = Matrix::new(vec![input])
            .dot(&self.weights)
            .add(&self.biases)
            .exp();
        totals
    }
}
