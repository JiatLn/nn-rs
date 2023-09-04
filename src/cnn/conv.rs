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
        if h < 2 || w < 2 {
            panic!("The input shape must be greater then 2x2");
        }
        self.last_input = input.clone();
        let mut output = (0..self.filter_num)
            .map(|_| Matrix::new_zero(h - 2, w - 2))
            .collect::<Vec<_>>();

        for i in 0..h - 2 {
            for j in 0..w - 2 {
                let slice_martix = input.slice(i, j, self.filter_size);
                for k in 0..self.filter_num {
                    let filter = &self.filters[k];
                    let sum = slice_martix.multiple_martix_sum(filter);
                    output[k].set(i, j, sum);
                }
            }
        }
        output
    }

    pub fn backprop(&mut self, d_l_d_out: &Vec<Matrix<f64>>, lr: f64) -> () {
        let mut d_l_d_filters = (0..self.filter_num)
            .map(|_| Matrix::new_zero(self.filter_size, self.filter_size))
            .collect::<Vec<_>>();

        let (h, w) = self.last_input.shape();

        for i in 0..h - 2 {
            for j in 0..w - 2 {
                let slice_martix = self.last_input.slice(i, j, self.filter_size);
                for k in 0..self.filter_num {
                    d_l_d_filters[k] += slice_martix.clone() * d_l_d_out[k].get(i, j);
                }
            }
        }

        self.filters = (0..self.filter_num)
            .map(|i| self.filters[i].clone() - d_l_d_filters[i].clone() * lr)
            .collect::<Vec<_>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_3x3() {
        let mut conv = ConvLayer::new(3, 8);

        assert_eq!(conv.filter_num, 8);
        assert_eq!(conv.filters[0].shape(), (3, 3));

        let input = Matrix::new(vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![5.0, 3.0, 6.0, 4.0],
            vec![1.0, 2.0, 4.0, 3.0],
            vec![6.0, 8.0, 7.0, 9.0],
        ]);
        let output = conv.forward(&input);

        assert_eq!(output.len(), 8);
        assert_eq!(output[0].shape(), (2, 2));
    }
}
