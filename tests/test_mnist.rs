use nn_rs::{load_mnist_data, Matrix};

#[test]
fn test_mnist() {
    let mnist = load_mnist_data();

    let train_data = Matrix::from_shape_vec(mnist.trn_img, (50_000, 28, 28));

    assert_eq!(train_data.len(), 50_000);
    assert_eq!(train_data[0].shape(), (28, 28));
}
