use crate::Neuron;

pub struct NeuronWork<T> {
    h1: Neuron<T>,
    h2: Neuron<T>,
    o1: Neuron<T>,
}

impl NeuronWork<f64> {
    pub fn new(weights: &[f64], bias: f64) -> Self {
        NeuronWork {
            h1: Neuron::new(weights.to_vec(), bias),
            h2: Neuron::new(weights.to_vec(), bias),
            o1: Neuron::new(weights.to_vec(), bias),
        }
    }
    pub fn feedforward(self, x: &[f64]) -> f64 {
        let out_h1 = self.h1.feedforward(x);
        let out_h2 = self.h2.feedforward(x);

        self.o1.feedforward(&[out_h1, out_h2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_network_feedforward() {
        let network = NeuronWork::new(&[0.0, 1.0], 0.0);
        let x = vec![2.0, 3.0];

        assert_eq!(network.feedforward(&x), 0.7216325609518421);
    }
}
