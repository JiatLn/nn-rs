use crate::Matrix;

pub struct SoftmaxLayer {
    pub weights: Matrix<f64>,
    pub biases: Matrix<f64>,
}

impl SoftmaxLayer {
    pub fn new(input_len: usize, nodes: usize) -> Self {
        let weights = Matrix::new_randn(input_len, nodes) / input_len as f64;
        let biases = Matrix::new_zero(1, nodes);
        SoftmaxLayer { weights, biases }
    }

    pub fn forward(&self, input: &Vec<Matrix<f64>>) -> Matrix<f64> {
        let input = Matrix::new(vec![input
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()]);

        let totals = input * &self.weights + &self.biases;
        totals.exp()
    }
}