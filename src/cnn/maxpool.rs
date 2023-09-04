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

        self.last_input = input.clone();

        (0..filter_num)
            .map(|filter_idx| max_pool(&input[filter_idx], self.size))
            .collect::<Vec<_>>()
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
                    let sm =
                        self.last_input[filter_idx].slice(i * self.size, j * self.size, self.size);

                    let max = sm.max();
                    let (h, w) = sm.shape();

                    for i2 in 0..h {
                        for j2 in 0..w {
                            for k in 0..n {
                                if sm.get(i2, j2) == max {
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

fn max_pool(input: &Matrix<f64>, size: usize) -> Matrix<f64> {
    let (h, w) = input.shape();

    if w % 2 != 0 || h % 2 != 0 {
        panic!("shape size must be even")
    }

    Matrix::new(
        (0..h / 2)
            .map(|i| {
                (0..w / 2)
                    .map(|j| input.slice(i * size, j * size, size).max())
                    .collect()
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
