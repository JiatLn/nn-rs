use nn_rs::{load_mnist_data, CNNConfig, Matrix, CNN};

#[test]
fn test_cnn() {
    let mnist = load_mnist_data();

    let test_images = Matrix::from_shape_vec(&mnist.tst_img, (10_000, 28, 28));

    let test_labels = mnist.tst_lbl.iter().map(|&i| i as usize).collect();

    assert_eq!(test_images.len(), 10_000);
    assert_eq!(test_images[0].shape(), (28, 28));

    let cnn = CNN::new(CNNConfig {
        filter_num: 8,
        filter_size: 3,
        pool_size: 2,
        input_len: 13 * 13 * 8,
        nodes: 10,
    });

    println!("MNIST CNN initialized!");

    cnn.run(&test_images[0..1000].to_vec(), &test_labels);
}
