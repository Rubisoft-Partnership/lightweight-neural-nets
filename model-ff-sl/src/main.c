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

const int input_size = DATA_FEATURES;
const int num_classes = DATA_CLASSES;
const int layers_sizes[] = {DATA_FEATURES, 500, 500, 500};

const int layers_number = sizeof(layers_sizes) / sizeof(layers_sizes[0]);

// Hyper Parameters.
double learning_rate = 0.005;
const double beta1 = 0.9;
const double beta2 = 0.999;
const int epochs = 5;
const int batch_size = 10;
const double threshold = 4.0;

Dataset data;
FFNet ffnet;

static void setup(void)
{
    // set_seed(time(NULL)); // comment for reproducibility
    set_log_level(LOG_DEBUG);
    open_log_file_with_timestamp();

    data = dataset_split();
    const Loss loss_suite = LOSS_FF;

    // Load the model from checkpoint file.
    // load_ff_net(&ffnet, "ffnet.bin", relu, pdrelu, beta1, beta2);

    // Build the model from scratch.
    ffnet = new_ff_net(layers_sizes, layers_number, relu, pdrelu, threshold, beta1, beta2, loss_suite);
}

static void train_loop(void)
{
    clock_t start_time = clock();
    // Since batch is used for all layers, sample size is set to the maximum of the layers sizes.
    FFBatch batch = new_ff_batch(batch_size, max_int(layers_sizes, layers_number));

    for (int i = 0; i < epochs; i++) // iterate over epochs
    {
        clock_t epoch_start_time = clock();
        printf("Epoch %d\n", i);
        log_info("Epoch %d", i);
        shuffle_data(data.train);
        double loss = 0.0f;
        int num_batches = data.train->rows / batch_size;
        for (int j = 0; j < num_batches; j++) // iterate over batches
        {
            generate_batch(data.train, j, batch); // generate positive and negative samples
            loss += train_ff_net(ffnet, batch, learning_rate);
        }
        printf("\tLoss %.12f\n", (double)loss / num_batches);
        double epoch_time = (double)(clock() - epoch_start_time) / CLOCKS_PER_SEC;
        printf("\tEpoch time: %.2f seconds\n", epoch_time);
    }
    double total_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("\nTotal training time: %.2f seconds\n\n", total_time);

    free_ff_batch(batch);
}

void evaluate(void)
{
    log_info("Testing FFNet...");
    init_predictions(num_classes);
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
        const int prediction = predict_ff_net(ffnet, input, num_classes, input_size);
        add_prediction(ground_truth, prediction);
    }
    printf("Accuracy: %.2f\n", get_accuracy());
    printf("Balanced accuracy: %.2f\n", get_balanced_accuracy());
    printf("\n");
    print_normalized_confusion_matrix();
    
    // Save the model to a checkpoint file.
    save_ff_net(ffnet, "ffnet.bin");
}

int main(void)
{
    setup();

    train_loop();
    log_info("Training done");

    printf("Testing...\n");
    evaluate();

    free_dataset(data);
    free_ff_net(ffnet);
    close_log_file();
    return 0;
}
