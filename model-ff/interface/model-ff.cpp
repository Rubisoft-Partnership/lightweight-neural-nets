#include "model-ff.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <unordered_map>

extern "C"
{
#include <ff-net/ff-net.h>
#include <logging/logging.h>
#include <utils/utils.h>
}

#define FF_LOG_DIR "logs/model-ff-logs"

#include <config/config.hpp>

// TODO: add parameter layer_epochs.

void ModelFF::build(const std::string &data_path)
{
    using namespace config;
    units = parameters::units;
    num_classes = parameters::num_classes;
    threshold = parameters::ff::threshold;
    beta1 = parameters::ff::beta1;
    beta2 = parameters::ff::beta2;
    loss = parameters::ff::loss;
    // Initialize the model with the given parameters.
    // Convert int vector to int array.
    int layers_num = units.size();
    int *units_array = new int[layers_num];
    std::copy(units.begin(), units.end(), units_array);

    auto seed = std::hash<std::string>{}(data_path);
    set_seed(seed); // comment for reproducibility
    set_log_level(LOG_DEBUG);
    // Check if the log directory exists, if not create it.
    if (!std::filesystem::exists(FF_LOG_DIR))
    {
        if (!std::filesystem::create_directories(FF_LOG_DIR))
        {
            spdlog::error("Failed to create log directory: {}", FF_LOG_DIR);
            exit(EXIT_FAILURE);
        }
    }
    open_log_file_with_timestamp(FF_LOG_DIR);

    // Build the model.
    ffnet = new_ff_net(units_array, layers_num, relu, pdrelu, threshold, beta1, beta2, loss);

    log_info("Initializing model with the following parameters:\n");
    log_info("\tThreshold: %.2f\n", threshold);
    log_info("\tLoss function: %d\n", loss);
    log_info("\tLayer units: ");
    for (int i = 0; i < layers_num; i++)
        log_info("\t%d ", units[i]);
    log_info("\n");

    // Initialize model data structure.
    spdlog::debug("Reading dataset from: {}", data_path);
    data = dataset_split(data_path.c_str(), num_classes);
    // Read the input size from the dataset compare to the selected input layer size.
    const int input_size = data.train->feature_len;
    if (units[0] != input_size)
    {
        log_error("Input size mismatch: %d != %d\n", units[0], input_size);
        exit(EXIT_FAILURE);
    }
    // Compute dataset size.
    dataset_size = data.train->rows;
}

void ModelFF::train(const int &epochs, const int &batch_size, const double &learning_rate, std::function<void()> on_enumerate_epoch)
{
    // Find max layer size.
    int max_units = *std::max_element(units.begin(), units.end());

    clock_t start_time = clock();
    // Since batch is used for all layers, sample size is set to the maximum of the layers sizes.
    FFBatch batch = new_ff_batch(batch_size, max_units);

    on_enumerate_epoch();
    for (int i = 0; i < epochs; i++) // iterate over model epochs
    {
        clock_t epoch_start_time = clock();
        printf("Epoch %d\n", i);
        log_info("Epoch %d", i);
        shuffle_data(data.train);
        double loss = 0.0f;
        int num_batches = data.train->rows / batch_size;
        // Print progress bar
        init_progress_bar();

        for (int j = 0; j < num_batches; j++) // iterate over batches
        {
            // Update progress bar
            update_progress_bar(j, num_batches);

            generate_batch(data.train, j, batch); // generate positive and negative samples
            loss += train_ff_net(ffnet, batch, learning_rate);
        }
        finish_progress_bar();
        printf("\tLoss %.12f\n", (double)loss / num_batches);
        int epoch_time = (clock() - epoch_start_time) / CLOCKS_PER_SEC;
        printf("\tEpoch time: ");
        print_elapsed_time(epoch_time);
        printf("\n\n");
        on_enumerate_epoch();
    }
    int total_time = (clock() - start_time) / CLOCKS_PER_SEC;
    printf("Total training time: ");
    print_elapsed_time(total_time);
    printf("\n\n");

    free_ff_batch(batch);
}

metrics::Metrics ModelFF::evaluate()
{
    log_info("Testing FFNet...");
    init_predictions();

    for (int i = 0; i < data.test->rows; i++)
    {
        double *const input = data.test->input[i];
        double *const target = data.test->target[i];
        int ground_truth = -1;

        for (int j = 0; j < data.test->num_class; j++)
        {
            if (target[j] == 1.0f)
            {
                ground_truth = j;
                break;
            }
        }

        const int prediction = predict_ff_net(ffnet, input, num_classes, units[0]);
        add_prediction(ground_truth, prediction);
    }

    // Create a Metrics object and generate the metrics
    metrics::Metrics metrics;
    metrics.generate();

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
    log_debug("FFNet saved to %s", filename.c_str());
}

void ModelFF::load(const std::string filename)
{
    load_ff_net(ffnet, filename.c_str(), relu, pdrelu, beta1, beta2, false); // set checkpoint default path to false
    log_debug("FFNet loaded from %s", filename.c_str());
}
