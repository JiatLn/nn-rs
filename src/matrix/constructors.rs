use crate::Matrix;

impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn from_vec(vec: &[T], h: usize, w: usize) -> Self {
        if h * w != vec.len() {
            panic!("shape not match!");
        }
        Matrix::new((0..h).map(|i| vec[i..i + w].to_vec()).collect())
    }
}

impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn from_shape_vec(vec: &Vec<T>, shape: (usize, usize, usize)) -> Vec<Self> {
        let (n, h, w) = shape;
        if vec.len() != n * h * w {
            panic!("shape not match!");
        }

        (0..n)
            .map(|i| {
                let start = i * h * w;
                let end = (i + 1) * h * w;
                Matrix::from_vec(&vec[start..end], h, w)
            })
            .collect()
    }
}
