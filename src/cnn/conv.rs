use crate::Matrix;

#[derive(Debug)]
pub struct ConvLayer {
    filter_size: usize,
    filter_num: usize,
    filters: Vec<Matrix<f64>>,
    last_input: Matrix<f64>,
}

impl ConvLayer {
    pub fn new(filter_size: usize, filter_num: usize) -> Self {
        let filters = (0..filter_num)
            .map(|_| {
                Matrix::new_randn(filter_size, filter_size) / (filter_size * filter_size) as f64
            })
            .collect();

        ConvLayer {
            filter_size,
            filter_num,
            filters,
            last_input: Matrix::default(),
        }
    }

    pub fn forward(&mut self, input: &Matrix<f64>) -> Vec<Matrix<f64>> {
        let (h, w) = input.shape();
        if h < self.filter_size || w < self.filter_size {
            panic!("The input shape must be greater then filter size");
        }

        self.last_input = input.clone();

        (0..self.filter_num)
            .map(|i| convolve(input, &self.filters[i]))
            .collect()
    }

    pub fn backprop(&mut self, d_l_d_out: &Vec<Matrix<f64>>, lr: f64) -> () {
        let mut d_l_d_filters = (0..self.filter_num)
            .map(|_| Matrix::new_zero(self.filter_size, self.filter_size))
            .collect::<Vec<_>>();

        let (h, w) = self.last_input.shape();

        for idx in 0..self.filter_num {
            for i in 0..h - self.filter_size + 1 {
                for j in 0..w - self.filter_size + 1 {
                    d_l_d_filters[idx] +=
                        self.last_input.slice(i, j, self.filter_size) * d_l_d_out[idx].get(i, j);
                }
            }
        }

        for i in 0..self.filter_num {
            self.filters[i] -= d_l_d_filters[i].clone() * lr;
        }
    }
}

fn convolve(input: &Matrix<f64>, filter: &Matrix<f64>) -> Matrix<f64> {
    let (h1, w1) = input.shape();
    let (h2, w2) = filter.shape();

    let mut output = Matrix::new_zero(h1 - h2 + 1, w1 - w2 + 1);

    let (h, w) = output.shape();

    for i in 0..h {
        for j in 0..w {
            let sm = input.slice(i, j, w2);
            let sum = sm.multiple_martix_sum(filter);
            output.set(i, j, sum);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_3x3() {
        let mut conv = ConvLayer::new(3, 1);

        assert_eq!(conv.filter_num, 1);
        assert_eq!(conv.filters[0].shape(), (3, 3));

        let input = Matrix::new(vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![5.0, 3.0, 6.0, 4.0],
            vec![1.0, 2.0, 4.0, 3.0],
            vec![6.0, 8.0, 7.0, 9.0],
        ]);
        let output = conv.forward(&input);

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].shape(), (2, 2));
    }

    #[test]
    fn test_conv_calc() {
        let input = Matrix::new(vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![5.0, 3.0, 6.0, 4.0],
            vec![1.0, 2.0, 4.0, 3.0],
        ]);

        let filter = Matrix::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]]);

        let output = convolve(&input, &filter);

        assert_eq!(output.shape(), (2, 3));
        assert_eq!(
            output.0,
            vec![vec![23.0, 19.0, 20.0], vec![15.0, 23.0, 25.0]]
        );
    }
}
