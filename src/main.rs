use nn_rs::{load_mnist_data, CNNConfig, Matrix, CNN};

fn main() {
    let mnist = load_mnist_data();

    let vec = mnist
        .trn_img
        .iter()
        .map(|&v| v as f64 / 256.0)
        .collect::<Vec<_>>();

    let train_images = Matrix::from_shape_vec(&vec, (50_000, 28, 28));

    let train_labels = mnist
        .trn_lbl
        .iter()
        .map(|&i| i as usize)
        .collect::<Vec<_>>();

    assert_eq!(train_images.len(), 50_000);
    assert_eq!(train_images[0].shape(), (28, 28));

    let mut cnn = CNN::new(CNNConfig {
        filter_size: 3,
        filter_num: 8,
        pool_size: 2,
        input_len: 13 * 13 * 8,
        nodes: 10,
    });

    println!("MNIST CNN initialized!");

    cnn.run(&train_images[..1000], &train_labels[..1000]);
}
