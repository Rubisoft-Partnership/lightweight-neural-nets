#include <iostream>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          const std::vector<int> &layer_units)
{
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

    using fc = tiny_dnn::layers::fc;
    using tanh = tiny_dnn::activation::tanh;
    using softmax = tiny_dnn::activation::softmax;

    // construct nets

    for (int i = 0; i < layer_units.size() - 2; i++)
    {
        nn << fc(layer_units[i], layer_units[i + 1], true, backend_type) << tanh();
    }
    nn << fc(layer_units[layer_units.size() - 2], layer_units.back(), true, backend_type) // F6, 200-in, 10-out
       << softmax();

}

static void train(const std::string &data_dir_path,
                  double learning_rate,
                  const int n_train_epochs,
                  const int n_minibatch,
                  const std::vector<int> &layer_units)
{
    // specify loss-function and learning strategy
    tiny_dnn::network<tiny_dnn::sequential> nn;
    tiny_dnn::adagrad optimizer;

    construct_net(nn, layer_units);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

    tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                                 &train_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                                 &train_images, -1.0, 1.0, 2, 2);
    tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                                 &test_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                                 &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start training" << std::endl;

    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;

    optimizer.alpha *=
        std::min(tiny_dnn::float_t(4),
                 static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]()
    {
        std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
                  << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]()
    { disp += n_minibatch; };

    // training
    nn.train<tiny_dnn::cross_entropy>(optimizer, train_images, train_labels, n_minibatch,
                                      n_train_epochs, on_enumerate_minibatch,
                                      on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);
    // save network model & trained weights
    nn.save("bp-model");
}