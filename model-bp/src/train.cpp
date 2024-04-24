#include <iostream>
#include <vector>

#include <tiny_dnn/tiny_dnn.h>

extern "C" {
    #include <confusion-matrix/confusion-matrix.h>
    #include <predictions/predictions.h>
    #include <accuracy/accuracy.h>
}

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          const std::vector<int> &layer_units)
{
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

    using fc = tiny_dnn::layers::fc;
    using tanh = tiny_dnn::activation::tanh;
    using softmax = tiny_dnn::activation::softmax;

    // construct nets
    nn = tiny_dnn::make_mlp<tanh>(layer_units.begin(), layer_units.end());
    nn << softmax();
}

void generate_metrics(tiny_dnn::result results)
{
    init_predictions();
    //itearate over confusion matrix
    for (int i = 0; i < results.confusion_matrix.size(); i++)
    {
        for (int j = 0; j < results.confusion_matrix[i].size(); j++)
        {
            for (int k = 0; k < results.confusion_matrix[i][j]; k++)
            {
                add_prediction(i, j);
            }
        }
    }
    //print confusion matrix
    print_normalized_confusion_matrix();
    //print accuracy
    std::cout << "Accuracy: " << get_accuracy() << std::endl;
    std::cout << "Balanced accuracy: " << get_balanced_accuracy() << std::endl;
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
                                 &train_images, -1.0, 1.0, 0, 0);
    tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                                 &test_labels);
    tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                                 &test_images, -1.0, 1.0, 0, 0);

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
    generate_metrics(nn.test(test_images, test_labels));
   
    // save network model & trained weights
    nn.save("models/bp-model");
}