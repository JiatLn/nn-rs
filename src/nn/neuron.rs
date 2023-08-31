use crate::utils::{dot, sigmoid};

pub struct Neuron<T> {
    pub weights: Vec<T>,
    pub bias: T,
}

impl<T> Neuron<T> {
    pub fn new(weights: Vec<T>, bias: T) -> Self {
        Neuron { weights, bias }
    }
}

impl Neuron<f64> {
    pub fn feedforward(self, inputs: Vec<f64>) -> f64 {
        if inputs.len() != self.weights.len() {
            return 0.0;
        }
        let total = dot(&self.weights, &inputs) + self.bias;
        sigmoid(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_feedforward() {
        let n = Neuron::new(vec![0.0, 1.0], 4.0);
        let x = vec![2.0, 3.0];

        assert_eq!(n.feedforward(x), 0.9990889488055994);
    }
}
