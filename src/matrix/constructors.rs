use crate::Matrix;

impl Matrix<f64> {
    pub fn from_shape_vec(vec: &Vec<u8>, shape: (usize, usize, usize)) -> Vec<Self> {
        if vec.len() != shape.0 * shape.1 * shape.2 {
            panic!("shape not match!");
        }
        (0..shape.0)
            .map(|i| {
                let start = i * shape.1 * shape.2;
                let end = (i + 1) * shape.1 * shape.2;
                let vec_f64 = vec[start..end]
                    .to_vec()
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>();
                let v = (0..shape.1)
                    .map(|i| {
                        let start = i * shape.1;
                        let end = (i + 1) * shape.1;
                        vec_f64[start..end].to_vec()
                    })
                    .collect();
                Matrix::new(v)
                // let slice = vec.splice(range, replace_with)
            })
            .collect()
    }
}
