/**
 * @file main.c
 * @brief This file contains the main function and related functions for training and evaluating a feedforward neural network.
 *
 * The main function sets up the necessary components, trains the neural network, and evaluates its performance.
 * The code includes the necessary header files and defines the input and output sizes, layers sizes, and hyperparameters.
 * It also defines the data structure for storing the input data and the neural network model.
 *
 * The setup function initializes the necessary components such as logging and data structures.
 *
 * The train_loop function performs the training loop, shuffling the data, generating samples, and updating the neural network weights.
 *
 * The evaluate function tests the trained neural network by making predictions on test data and calculating the confusion matrix.
 *
 * The main function calls the setup function, performs the training loop, calls the evaluate function, and cleans up the resources.
 *
 * @note This code assumes that the necessary header files and libraries are available.
 * @note The input and output sizes, layers sizes, and hyperparameters are hard-coded input this code.
 * @note The code assumes that the data structure and functions for data manipulation, logging, losses, and the confusion matrix are available.
 */
#include <time.h>
#include <stdlib.h>

#include <ff-net/ff-net.h>
#include <data/data.h>
#include <logging/logging.h>
#include <utils/utils.h>
#include <losses/losses.h>

#include <confusion-matrix/confusion-matrix.h>
#include <accuracy/accuracy.h>
#include <predictions/predictions.h>

// Input and output size is harded coded here as machine learning
// repositories usually don't include the input and output size input the data itself.
const int input_size = DATA_FEATURES;
const int num_classes = DATA_CLASSES;
const int layers_sizes[] = {DATA_FEATURES, 500};

const int layers_number = sizeof(layers_sizes) / sizeof(layers_sizes[0]);

// Hyper Parameters.
double learning_rate = 0.001;
const double beta1 = 0.9;
const double beta2 = 0.999;
const int epochs = 5;
const double threshold = 4.0;

Data data;
FFNet ffnet;

static void setup(void)
{
    // Comment to repeat the same results
    // set_seed(time(NULL));
    set_log_level(LOG_DEBUG);
    open_log_file_with_timestamp();

    data = data_build();
    const Loss loss_suite = LOSS_FF;
    ffnet = new_ff_net(layers_sizes, layers_number, relu, pdrelu, threshold, beta1, beta2, loss_suite);
    log_debug("FFNet built with the following layers:");
    increase_indent();
    for (int i = 0; i < ffnet.num_cells; i++)
        log_debug("Layer %d: %d inputs, %d outputs", i, ffnet.layers[i].input_size, ffnet.layers[i].output_size);
    decrease_indent();
}

static void train_loop(void)
{
    clock_t start_time = clock();
    FFsamples samples = new_ff_samples(input_size);
    for (int i = 0; i < epochs; i++)
    {
        clock_t epoch_start_time = clock();
        printf("Epoch %d\n", i);
        log_info("Epoch %d", i);
        increase_indent();
        shuffle_data(data);
        double loss = 0.0f;
        for (int j = 0; j < data.rows; j++)
        {
            generate_samples(data, j, samples);
            loss = train_ff_net(ffnet, samples.pos, samples.neg, learning_rate);
        }
        printf("\tLoss %.12f\n", (double)loss);
        double epoch_time = (double)(clock() - epoch_start_time) / CLOCKS_PER_SEC;
        printf("\tEpoch time: %.2f seconds\n", epoch_time);
        decrease_indent();
    }
    double total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("\nTotal training time: %.2f seconds\n\n", total_time);
    free_ff_samples(samples);
}

void evaluate(void)
{
    log_info("Testing FFNet...");
    initPredictions(num_classes);
    for (int i = 0; i < 100; i++)
    {
        double *const input = data.input[i];
        double *const target = data.target[i];
        int ground_truth = -1;
        for (int j = 0; j < data.num_class; j++)
        {
            if (target[j] == 1.0f)
            {
                ground_truth = j;
                break;
            }
        }
        const int prediction = predict_ff_net(ffnet, input, num_classes, input_size);
        addPrediction(ground_truth, prediction);
        addPredictionAccuracy(ground_truth, prediction);
    }
    printf("Accuracy: %.2f\n", getAccuracy());
    printf("\n");
    printNormalizedConfusionMatrix();
}

int main(void)
{
    setup();

    train_loop();
    log_info("Training done");

    printf("Testing...\n");
    evaluate();

    free_data(data);
    close_log_file();
    free_ff_net(ffnet);

    return 0;
}
