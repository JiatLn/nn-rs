use nn_rs::{load_mnist_data, ConvLayer, Matrix, MaxPoolLayer};

#[test]
fn test_maxpool() {
    let mnist = load_mnist_data();

    let train_data = Matrix::from_shape_vec(&mnist.trn_img, (50_000, 28, 28));

    assert_eq!(train_data.len(), 50_000);
    assert_eq!(train_data[0].shape(), (28, 28));

    let mut conv = ConvLayer::new(3, 8);
    let output = conv.forward(&train_data[0]);

    assert_eq!(output.len(), 8);
    assert_eq!(output[0].shape(), (26, 26));

    let mut maxpool = MaxPoolLayer::new(2);
    let output = maxpool.forward(&output);

    assert_eq!(output.len(), 8);
    assert_eq!(output[0].shape(), (13, 13));
}
