/// f(x) = 1 / (1 + (-x) ^ e)
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid
///
/// f'(x) = f(x) - (1 - f(x))
pub fn deriv_sigmoid(x: f64) -> f64 {
    let fx = sigmoid(x);
    fx * (1.0 - fx)
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).rfold(0.0, |v, (&a, &b)| v + a * b)
}

pub fn mse_loss(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred)
        .rfold(0.0, |v, (&y1, &y2)| v + (y1 - y2).powf(2.0))
        / y_true.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let y_true = vec![1.0, 0.0, 0.0, 1.0];
        let y_pred = vec![0.0, 0.0, 0.0, 0.0];

        assert_eq!(mse_loss(&y_true, &y_pred), 0.5);
    }
}
