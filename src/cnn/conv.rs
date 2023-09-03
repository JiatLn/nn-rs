use crate::Matrix;

#[derive(Debug)]
pub struct ConvLayer {
    filter_size: usize,
    filter_num: usize,
    filters: Vec<Matrix<f64>>,
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
        }
    }

    pub fn forward(&self, input: &Matrix<f64>) -> Vec<Matrix<f64>> {
        let (h, w) = input.shape();
        if h < 2 || w < 2 {
            panic!("The input shape must be greater then 2x2");
        }
        let mut output: Vec<_> = (0..self.filter_num)
            .map(|_| Matrix::new_zero(h - 2, w - 2))
            .collect();

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_3x3() {
        let conv = ConvLayer::new(3, 8);

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
