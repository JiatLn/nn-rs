use crate::Matrix;

pub struct MaxPoolLayer {
    size: usize,
}

impl MaxPoolLayer {
    pub fn new(size: usize) -> Self {
        MaxPoolLayer { size }
    }
    pub fn forward(self, input: &Vec<Matrix<f64>>) -> Vec<Matrix<f64>> {
        let filter_num = input.len();
        let (h, w) = input[0].shape();

        if w % 2 != 0 || h % 2 != 0 {
            panic!("weight & height size must be even")
        }

        let mut output = (0..filter_num)
            .map(|_| Matrix::new_zero(w / 2, h / 2))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_maxpool_2x2() {
        let maxpool = MaxPoolLayer::new(2);

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
