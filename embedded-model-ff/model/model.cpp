#include "model.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <unordered_map>

extern "C"
{
#include <ff-net/ff-net.h>
#include <logging/logging.h>
#include <utils/utils.h>
#include <losses/losses.h>
}

#define NUM_CLASSES 10
#define THRESHOLD 5.0
#define BETA1 0.9
#define BETA2 0.999

void ModelFF::build()
{
    LossType loss = LossType::LOSS_TYPE_FF;
    // Initialize the model with the given parameters.
    // Convert int vector to int array.

    printf("Building model...\n");

    int units_array[] = {100, 50};
#define LAYERS_NUM 2
    // Bigger layer size
    max_units = *std::max_element(units_array, units_array + LAYERS_NUM);
    input_size = units_array[0];

    printf("Max units: %d\n", max_units);
    set_seed(time(NULL));

    printf("Building network...\n");

    // Build the model.
    ffnet = new_ff_net(units_array, LAYERS_NUM, relu, pdrelu, THRESHOLD, BETA1, BETA2, loss);

    printf("Network built.\n");
}

void ModelFF::train(const int &epochs, const int &batch_size, const double &learning_rate)
{

    FFBatch batch = new_ff_batch(batch_size, max_units);

    for (int i = 0; i < epochs; i++) // iterate over model epochs
    {
        shuffle_data(data.train);
        double loss = 0.0f;
        int num_batches = data.train->rows / batch_size;
        for (int j = 0; j < num_batches; j++) // iterate over batches
        {
            generate_batch(data.train, j, batch);                            // generate positive and negative samples
            loss += train_ff_net(ffnet, batch, learning_rate) / num_batches; // train the model
        }
    }
    free_ff_batch(batch);
}

metrics::Metrics ModelFF::evaluate()
{
    // Create a Metrics object and generate the metrics
    metrics::Metrics metrics;
    Predictions predictions;
    init_predictions(&predictions);
    metrics.loss = test_ff_net(ffnet, data.test, input_size, &predictions);
    metrics.generate(&predictions);

    return metrics;
}

std::vector<double> ModelFF::get_weights() const
{
    std::vector<double> weights;
    for (int i = 0; i < ffnet->num_cells; i++)                 // iterate over cells
        for (int j = 0; j < ffnet->layers[i].num_weights; j++) // iteratet over weights
            weights.push_back(ffnet->layers[i].weights[j]);    // add weight to vector
    return weights;
}

void ModelFF::set_weights(const std::vector<double> &weights)
{
    int weight_index = 0;
    for (int i = 0; i < ffnet->num_cells; i++)                     // iterate over cells
        for (int j = 0; j < ffnet->layers[i].num_weights; j++)     // iterate over weights
            ffnet->layers[i].weights[j] = weights[weight_index++]; // set weight from vector
}

void ModelFF::save(const std::string filename)
{
    save_ff_net(ffnet, filename.c_str(), false); // set checkpoint default path to false
}

void ModelFF::load(const std::string filename)
{
    load_ff_net(ffnet, filename.c_str(), relu, pdrelu, BETA1, BETA2, false); // set checkpoint default path to false
}
