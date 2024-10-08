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

#include <metrics.h>

// Default dataset parameters and model architecture.
char *dataset_path = "../dataset/digits/";
int num_classes = 10;
int input_size = 74;
int layers_sizes[20] = {74, 500, 500, 500};
int layers_number = 4;

// Hyper Parameters.
double learning_rate = 0.01;
const double beta1 = 0.9;
const double beta2 = 0.999;
int epochs = 5;
int batch_size = 10;
double threshold = 4.0;

Dataset data;
FFNet *ffnet;

void evaluate(void);
void parse_args(int argc, char **argv);


Metrics metrics;

static void setup(void)
{
    // set_seed(time(NULL)); // comment for reproducibility
    set_log_level(LOG_DEBUG);
    open_log_file_with_timestamp("logs");

    data = dataset_split(dataset_path, num_classes);
    // Read the input size from the dataset and set the first layer size.
    input_size = data.train->feature_len;
    layers_sizes[0] = input_size;

    // Load the model from checkpoint file.
    // load_ff_net(&ffnet, "ffnet.bin", relu, pdrelu, beta1, beta2, true);

    // Build the model from scratch.
    ffnet = new_ff_net(layers_sizes, layers_number, relu, pdrelu, threshold, beta1, beta2, LOSS_TYPE_FF);

    printf("Running with the following parameters:\n");
    printf("\tDataset path: %s\n", dataset_path);
    printf("\tLearning rate: %.4f\n", learning_rate);
    printf("\tEpochs: %d\n", epochs);
    printf("\tBatch size: %d\n", batch_size);
    printf("\tThreshold: %.2f\n", threshold);
    printf("\tLayer units: ");
    for (int i = 0; i < layers_number; i++)
    {
        printf("%d ", layers_sizes[i]);
    }
    printf("\n\n");
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
        evaluate();
    }
    int total_time = (clock() - start_time) / CLOCKS_PER_SEC;
    printf("Total training time: ");
    print_elapsed_time(total_time);
    printf("\n\n");

    free_ff_batch(batch);
}

void evaluate(void)
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
        const int prediction = predict_ff_net(ffnet, input, num_classes, input_size);
        add_prediction(ground_truth, prediction);
    }
    reset_metrics(metrics);
    metrics = generate_metrics();
    print_metrics(metrics);

    // Save the model to a checkpoint file.
    save_ff_net(ffnet, "ffnet.bin", true);
    log_debug("FFNet saved to ffnet.bin");
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);
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

void parse_args(int argc, char **argv)
{
    if (argc == 1)
        return;
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
        printf("Usage: %s [OPTIONS]\n", argv[0]);
        printf("Options:\n");
        printf("  -lr, --learning_rate\tLearning rate for the optimizer (default: %.4f)\n", learning_rate);
        printf("  -e,  --epochs\t\tNumber of epochs for training (default: %d)\n", epochs);
        printf("  -bs, --batch_size\tBatch size for training (default: %d)\n", batch_size);
        printf("  -t,  --threshold\tThreshold for the activation function (default: %.2f)\n", threshold);
        printf("  -lu, --layer_units\tWidth of each layer (default: ");
        for (int i = 0; i < layers_number; i++)
        {
            printf("%d ", layers_sizes[i]);
        }
        printf(")\n");
        printf("  -dp, --dataset_path\tPath to the dataset (default: %s)\n", dataset_path);
        exit(0);
    }
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-lr") == 0 || strcmp(argv[i], "--learning_rate") == 0)
        {
            learning_rate = atof(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--epochs") == 0)
        {
            epochs = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-bs") == 0 || strcmp(argv[i], "--batch_size") == 0)
        {
            batch_size = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threshold") == 0)
        {
            threshold = atof(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-lu") == 0 || strcmp(argv[i], "--layer_units") == 0)
        {
            layers_number = 0;
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                layers_sizes[layers_number] = atoi(argv[i + 1]);
                layers_number++;
                i++;
            }
        }
        else if (strcmp(argv[i], "-dp") == 0 || strcmp(argv[i], "--dataset_path") == 0)
        {
            dataset_path = argv[i + 1];
            i++;
        }
        else
        {
            log_error("Unknown option: %s", argv[i]);
            exit(1);
        }
    }
}