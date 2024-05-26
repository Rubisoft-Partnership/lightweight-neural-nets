#include "model-bp.hpp"

void ModelBP::build(const std::vector<int> &units, const std::string &data_path)
{
    // Initialize the model
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
    using fc = tiny_dnn::layers::fc;
    using relu = tiny_dnn::activation::relu;
    using softmax = tiny_dnn::activation::softmax;

    bpnet = tiny_dnn::make_mlp<relu>(units.begin(), units.end());
    bpnet << softmax();

    // Load MNIST dataset
    tiny_dnn::parse_mnist_labels(data_path + "/train-labels.idx1-ubyte", &train_labels);
    tiny_dnn::parse_mnist_images(data_path + "/train-images.idx3-ubyte", &train_images, min_scale, max_scale, x_padding, y_padding);
    tiny_dnn::parse_mnist_labels(data_path + "/t10k-labels.idx1-ubyte", &test_labels);
    tiny_dnn::parse_mnist_images(data_path + "/t10k-images.idx3-ubyte", &test_images, min_scale, max_scale, x_padding, y_padding);
}

void ModelBP::train(const int &epochs, const int &batch_size, const double &learning_rate)
{
    // Specify loss-function and learning strategy
    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;
    optimizer.alpha *= std::min(tiny_dnn::float_t(4), static_cast<tiny_dnn::float_t>(sqrt(batch_size) * learning_rate));

    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]()
    {
        std::cout << "Epoch " << epoch << "/" << epochs << " finished. "
                  << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        tiny_dnn::result res = bpnet.test(test_images, test_labels);

        disp.restart(train_images.size());
        t.restart();
    };
    auto on_enumerate_minibatch = [&]()
    { disp += batch_size; };

    // Training
    bpnet.train<tiny_dnn::cross_entropy>(optimizer, train_images, train_labels, batch_size, epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "Training finished. It took " << t.elapsed() << " seconds." << std::endl;
}

metrics::Metrics ModelBP::evaluate()
{
    tiny_dnn::result results = bpnet.test(test_images, test_labels);

    init_predictions();
    // iterate over confusion matrix
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

    // Create a Metrics object and generate the metrics
    metrics::Metrics metrics;
    metrics.generate();

    // Print the metrics
    metrics.print();
    return metrics;
}

std::vector<double> ModelBP::get_weights() const
{
    std::vector<double> weights;
    for (int i = 0; i < bpnet.layer_size(); i++)
    {
        const std::vector<const tiny_dnn::vec_t *>& layer_weights = bpnet[i]->weights();
        for (int j = 0; j < layer_weights.size(); j++)
            for (int k = 0; k < layer_weights[j]->size(); k++)
                weights.push_back(layer_weights[j]->at(k));
    }
    return weights;
}

void ModelBP::set_weights(const std::vector<double> &weights)
{
    int weight_index = 0;
    for (int i = 0; i < bpnet.layer_size(); i++)
    {
        const std::vector<tiny_dnn::vec_t *>& layer_weights = bpnet[i]->weights();
        for (int j = 0; j < layer_weights.size(); j++)
            for (int k = 0; k < layer_weights[j]->size(); k++)
                bpnet[i]->weights()[j]->at(k) = weights[weight_index++];
    }
}

void ModelBP::save(const std::string filename)
{
    bpnet.save(filename);
}

void ModelBP::load(const std::string filename)
{
    bpnet.load(filename);
}