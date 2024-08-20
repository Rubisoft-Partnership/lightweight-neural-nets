#include "model.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <unordered_map>

#include "esp_timer.h"
#include "esp_task_wdt.h"

extern "C"
{
#include <ff-net/ff-net.h>
#include <utils/utils.h>
#include <losses/losses.h>
}

#include "data.hpp"


#define THRESHOLD 5.0
#define BETA1 0.9
#define BETA2 0.999

void generate_negative_samples(FFBatch batch, int sample_size)
{
    for (int i = 0; i < batch.size; i++)
    {
        int lab_idx = 0;
        while (lab_idx - 10 < 0)
        {
            if (batch.pos[i][sample_size - 10 + lab_idx] == 1)
                break;
        }
        printf("Positive label index: %d\n", lab_idx);
        int step = 1 + get_random() % (10 - 1);
        int neg_label = (lab_idx + step) % 10;
        printf("Negative label index: %d\n", neg_label);
        batch.neg[i][sample_size - 10 + neg_label] = 1;
        batch.neg[i][sample_size - 10 + lab_idx] = 0;
        esp_task_wdt_reset();
    }
}

void ModelFF::build()
{
    LossType loss = LossType::LOSS_TYPE_FF;
    // Initialize the model with the given parameters.
    // Convert int vector to int array.

    printf("Building model...\n");

    int units_array[] = {FEATURES, 50};
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
    esp_task_wdt_add(NULL);
    FFBatch batch = new_ff_batch(batch_size, max_units);

    int sample_index = 0;
    for(int i = 0; i < batch_size; i++)
    {
        if (sample_index >= NUM_SAMPLES)
            sample_index = 0;
        memcpy(batch.pos[i], samples[sample_index], FEATURES);
        memcpy(batch.neg[i], samples[sample_index], FEATURES);
        sample_index++;
    }

    esp_task_wdt_reset();

    printf("Generating negative samples...\n");
    generate_negative_samples(batch, FEATURES);

    for (int i = 0; i < epochs; i++) // iterate over model epochs
    {
        double loss = 0.0f;
        int64_t start_time = esp_timer_get_time();
        loss += train_ff_net(ffnet, batch, learning_rate); // train the model
        int64_t end_time = esp_timer_get_time();
        double elapsed_time = (end_time - start_time) / 1000.0;
        printf("Training completed in %.2f ms\n", elapsed_time);
        printf("Epoch %d: Loss: %f\n", i, loss);
    }
    printf("Training complete.\n");
    free_ff_batch(batch);
    esp_task_wdt_delete(NULL);
}

// metrics::Metrics ModelFF::evaluate()
// {
//     // Create a Metrics object and generate the metrics
//     metrics::Metrics metrics;
//     Predictions predictions;
//     init_predictions(&predictions);
//     metrics.loss = test_ff_net(ffnet, data.test, input_size, &predictions);
//     metrics.generate(&predictions);

//     return metrics;
// }

