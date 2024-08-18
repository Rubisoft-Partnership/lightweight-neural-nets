#include "model.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <unordered_map>

#include "driver/uart.h"

extern "C"
{
#include <ff-net/ff-net.h>
#include <logging/logging.h>
#include <utils/utils.h>
#include <losses/losses.h>
}

#define UART_NUM UART_NUM_0
#define BUF_SIZE (1024)

#define NUM_CLASSES 10
#define THRESHOLD 5.0
#define BETA1 0.9
#define BETA2 0.999

// void generate_negative_samples(FFBatch batch);

void ModelFF::build()
{
    LossType loss = LossType::LOSS_TYPE_FF;
    // Initialize the model with the given parameters.
    // Convert int vector to int array.

    printf("Building model...\n");

    int units_array[] = {74, 50};
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

    char serial_buffer[BUF_SIZE];
    sprintf(serial_buffer, "READY. BS: %d\n", batch_size);
    int len = strlen(serial_buffer);
    uart_write_bytes(UART_NUM, (const char *)serial_buffer, len);

    int feature_index = 0;
    int sample_num = 0;

    while (1)
    {
        // Read data from the UART
        int len = uart_read_bytes(UART_NUM, serial_buffer, BUF_SIZE - 1, 20 / portTICK_PERIOD_MS);
        if (len > 0)
        {
            serial_buffer[len] = '\0'; // Null-terminate the string
            printf("Received data: %s\n", serial_buffer);

            // Handle completion signal
            if (strcmp(serial_buffer, "DONE") == 0)
            {
                printf("Transfer complete\n");
                printf("Feature index: %d\n", feature_index);
                break;
            }

            // Convert the received string to a float and store it in the features array
            double received_value;
            sscanf(serial_buffer, "%lf", &received_value);
            batch.pos[sample_num][feature_index++] = received_value;

            // Ensure we do not exceed the features array size
            if (feature_index >= 74)
            {
                printf("ESP-DONE.\n");
                feature_index = 0;
                memcpy(batch.neg[sample_num], batch.pos[sample_num], 74);
                sample_num++;
                if (sample_num >= batch_size)
                {
                    printf("Batch full.\n");
                    break;
                }
            }
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }

    for (int i = 0; i < epochs; i++) // iterate over model epochs
    {
        double loss = 0.0f;
        loss += train_ff_net(ffnet, batch, learning_rate); // train the model
        printf("Epoch %d: Loss: %f\n", i, loss);
    }
    printf("Training complete.\n");
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
