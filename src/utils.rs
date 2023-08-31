pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        panic!("a.len != b.len");
    }
    a.iter().zip(b).rfold(0.0, |v, (&a, &b)| v + a * b)
}
