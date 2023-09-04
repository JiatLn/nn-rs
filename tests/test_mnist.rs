use nn_rs::{load_mnist_data, Matrix};

#[test]
fn test_mnist() {
    let mnist = load_mnist_data();

    let vec = mnist
        .trn_img
        .iter()
        .map(|&v| v as f64 / 256.0)
        .collect::<Vec<_>>();

    let train_images = Matrix::from_shape_vec(&vec, (50_000, 28, 28));

    assert_eq!(train_images.len(), 50_000);
    assert_eq!(train_images[0].shape(), (28, 28));
}
