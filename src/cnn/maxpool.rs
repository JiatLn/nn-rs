use crate::Matrix;

pub struct MaxPoolLayer {
    size: usize,
    last_input: Vec<Matrix<f64>>,
}

impl MaxPoolLayer {
    pub fn new(size: usize) -> Self {
        MaxPoolLayer {
            size,
            last_input: vec![],
        }
    }
    pub fn forward(&mut self, input: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        let filter_num = input.len();
        let (h, w) = input[0].shape();

        self.last_input = input.clone();

        if w % 2 != 0 || h % 2 != 0 {
            panic!("weight & height size must be even")
        }

        let mut output = (0..filter_num)
            .map(|_| Matrix::new_zero(h / 2, w / 2))
            .collect::<Vec<_>>();

        for filter_idx in 0..filter_num {
            for i in 0..h / 2 {
                for j in 0..w / 2 {
                    let slice_martix =
                        input[filter_idx].slice(i * self.size, j * self.size, self.size);
                    for k in 0..filter_num {
                        output[k].set(i, j, slice_martix.max());
                    }
                }
            }
        }
        output
    }

    pub fn backprop(&mut self, d_l_d_out: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        // d_L_d_input = np.zeros(self.last_input.shape)
        let n = self.last_input.len();
        assert!(n > 0);
        let (h, w) = self.last_input[0].shape();

        let mut d_l_d_input = (0..n).map(|_| Matrix::new_zero(h, w)).collect::<Vec<_>>();

        for filter_idx in 0..n {
            for i in 0..h / 2 {
                for j in 0..w / 2 {
                    let slice_martix =
                        self.last_input[filter_idx].slice(i * self.size, j * self.size, self.size);

                    let max = slice_martix.max();
                    let (h, w) = slice_martix.shape();
                    for i2 in 0..h {
                        for j2 in 0..w {
                            for k in 0..n {
                                if slice_martix.get(i2, j2) == max {
                                    d_l_d_input[k].set(
                                        i * 2 + i2,
                                        j * 2 + j2,
                                        d_l_d_out[k].get(i, j),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        d_l_d_input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_maxpool_2x2() {
        let mut maxpool = MaxPoolLayer::new(2);

        assert_eq!(maxpool.size, 2);

        let input = Matrix::new(vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![5.0, 3.0, 6.0, 4.0],
            vec![1.0, 2.0, 4.0, 3.0],
            vec![6.0, 8.0, 7.0, 9.0],
        ]);

        let output = maxpool.forward(&vec![input]);

        assert_eq!(output[0], Matrix::new(vec![vec![5.0, 6.0], vec![8.0, 9.0]]));
    }
}
