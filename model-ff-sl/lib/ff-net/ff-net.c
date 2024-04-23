/**
 * @file ff-net.c
 * @brief This file contains the implementation of a FFNet for the FF algorithm.
 *
 * */

#include <ff-net/ff-net.h>

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#include <logging/logging.h>
#include <ff-cell/ff-cell.h>
#include <ff-utils/ff-utils.h>

// Buffer to store output activations.
#define H_BUFFER_SIZE 1024
double o_buffer[H_BUFFER_SIZE]; // outputs buffer for positive pass

/**
 * @brief Builds a FFNet by creating multiple FFCell objects.
 *
 * This function constructs a feedforward neural network (FFNet) by creating multiple FFCell objects.
 * The FFNet is built based on the provided layer sizes, activation function, derivative of the activation function,
 * threshold value, beta1, beta2, and loss function suite.
 *
 * @param layer_sizes The array of layer sizes, including the number of input and output units.
 * @param num_layers The number of layers in the FFNet.
 * @param act The activation function for the FFNet.
 * @param pdact The derivative of the activation function for the FFNet.
 * @param treshold The threshold value for the FFNet.
 * @param beta1 The beta1 value of the Adam optimizer\.
 * @param beta2 The beta2 value of the Adam optimizer\.
 * @param loss_suite The loss function suite for the FFNet.
 * @return FFNet The constructed FFNet.
 */
FFNet new_ff_net(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double treshold, const double beta1, const double beta2, Loss loss_suite)
{
    FFNet ffnet;
    ffnet.loss_suite = loss_suite;
    ffnet.num_cells = num_layers - 1;
    ffnet.threshold = treshold;

    log_info("Building FFNet with %d layers, %d ff cells", num_layers, ffnet.num_cells);
    char layers_str[256];
    layers_str[0] = '\0';
    for (int i = 0; i < num_layers; i++)
    {
        char layer_str[32];
        snprintf(layer_str, sizeof(layer_str), "%d ", layer_sizes[i]);
        strcat(layers_str, layer_str);
    }
    log_info("Layers: %s", layers_str);

    for (int i = 0; i < ffnet.num_cells; i++)
        ffnet.layers[i] = new_ff_cell(layer_sizes[i], layer_sizes[i + 1], act, pdact, beta1, beta2);
    
    log_info("FFNet built with %d layers", ffnet.num_cells);
    return ffnet;
}

/**
 * @brief Frees the memory of a FFNet.
 *
 * @param ffnet The FFNet to free.
 */
void free_ff_net(FFNet ffnet)
{
    for (int i = 0; i < ffnet.num_cells; i++)
        free_ff_cell(ffnet.layers[i]);
}

/**
 * @brief Trains a FFNet by training each cell.
 *
 * @param ffnet The FFNet to train.
 * @param pos The positive training data.
 * @param neg The negative training data.
 * @param learning_rate The learning rate for the training.
 * @return double The total loss of the FFNet.
 */
double train_ff_net(const FFNet ffnet, const double *const pos, const double *const neg, const double learning_rate)
{
    double loss = 0.0;
    loss += train_ff_cell(ffnet.layers[0], pos, neg, learning_rate, ffnet.threshold, ffnet.loss_suite);
    for (int i = 1; i < ffnet.num_cells; i++)
        loss += train_ff_cell(ffnet.layers[i], o_buffer, ffnet.layers[i - 1].output, learning_rate, ffnet.threshold, ffnet.loss_suite);
    return loss;
}

/**
 * @brief Inference function for FFNet.
 *
 * @param ffnet The FFNet to perform inference on.
 * @param input The input data.
 * @param num_classes The number of classes.
 * @param input_size The size of the input data.
 * @return int The predicted class index.
 */
int predict_ff_net(const FFNet ffnet, const double *input, const int num_classes, const int input_size)
{
    log_debug("Predicting sample on model with cells: %d", ffnet.num_cells);
    double *netinput = (double *)malloc((input_size) * sizeof(double));
    double goodnesses[MAX_CLASSES];
    // For debugging.
    for (int i = 0; i < num_classes; i++)
        goodnesses[i] = 0.0;

    // For each class.
    for (int label = 0; label < num_classes; label++)
    {
        embed_label(netinput, input, label, input_size, num_classes);
        for (int i = 0; i < ffnet.num_cells; i++)
        {
            fprop_ff_cell(ffnet.layers[i], i == 0 ? netinput : ffnet.layers[i - 1].output);
            goodnesses[label] += goodness(ffnet.layers[i].output, ffnet.layers[i].output_size);
            normalize_vector(ffnet.layers[i].output, ffnet.layers[i].output_size);
            log_debug("Forward propagated label %d to network cell %d with cumulative goodness: %f", label, i, goodnesses[label]);
        }
    }

    free(netinput);

    int max_goodness_index = 0;
    for (int i = 1; i < num_classes; i++)
    {
        if (goodnesses[i] > goodnesses[max_goodness_index])
            max_goodness_index = i;
    }
    return max_goodness_index;
}

/**
 * @brief Saves a FFNet to a file.
 *
 * @param ffnet The FFNet to save.
 * @param filename The name of the file to save the FFNet.
 */
void save_ff_net(const FFNet ffnet, const char *filename)
{
    // Create the checkpoint directory if it does not exist
    if (access(FFNET_CHECKPOINT_PATH, F_OK) == -1)
    {
        if (mkdir(FFNET_CHECKPOINT_PATH, 0777) == -1)
        {
            log_error("Could not create checkpoint directory %s", FFNET_CHECKPOINT_PATH);
            return;
        }
    }

    FILE *file = NULL;
    char full_path[256];

    if (filename == NULL) // no filename provided
    {
        // Get the current time
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);

        // Create checkpoint filename
        char checkpoint_filename[256];
        strftime(checkpoint_filename, sizeof(checkpoint_filename), "%Y-%m-%d_%H-%M-%S", tm_info);

        // Construct the full path
        char full_path[256];
        snprintf(full_path, sizeof(full_path), "%s/checkpoint_%s.bin", FFNET_CHECKPOINT_PATH, checkpoint_filename);

        log_info("No filename provided, saving FFNet to file %s", full_path);
    }
    else
    {
        snprintf(full_path, sizeof(full_path), "%s/%s", FFNET_CHECKPOINT_PATH, filename);
        log_info("Saving FFNet to file %s", full_path);
    }

    file = fopen(full_path, "wb");
    if (file == NULL)
    {
        log_error("Could not open file %s for writing", full_path);
        return;
    }

    fwrite(&ffnet.num_cells, sizeof(ffnet.num_cells), 1, file);
    fwrite(&ffnet.threshold, sizeof(ffnet.threshold), 1, file);
    fwrite(&ffnet.loss_suite.type, sizeof(ffnet.loss_suite.type), 1, file);

    for (int i = 0; i < ffnet.num_cells; i++)
        save_ff_cell(ffnet.layers[i], file);

    fclose(file);
    log_info("Saved FFNet to file %s", filename);
}

/**
 * @brief Loads a FFNet from a file.
 *
 * @param ffnet The FFNet to load.
 * @param filename The name of the file to load the FFNet.
 * @param act The activation function.
 * @param pdact The derivative of the activation function.
 * @param beta1 The hyperparameter for the FF algorithm.
 * @param beta2 The hyperparameter for the FF algorithm.
 */
void load_ff_net(FFNet *ffnet, const char *filename, double (*act)(double), double (*pdact)(double),
                 const double beta1, const double beta2)
{
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "%s/%s", FFNET_CHECKPOINT_PATH, filename);

    log_debug("Loading FFNet from file %s", full_path);
    FILE *file = fopen(full_path, "rb");
    if (file == NULL)
    {
        log_error("Could not open file %s for reading", full_path);
        return;
    }

    // Read the FFNet number of cells, threshold and loss function type.
    fread(&ffnet->num_cells, sizeof(ffnet->num_cells), 1, file);
    fread(&ffnet->threshold, sizeof(ffnet->threshold), 1, file);
    enum LossType loss_type;
    fread(&loss_type, sizeof(loss_type), 1, file);

    log_debug("FFNet has %d cells, threshold %f and loss function type %d", ffnet->num_cells, ffnet->threshold, loss_type);

    switch (loss_type) // set the loss function
    {
    case LOSS_FF_TYPE:
        ffnet->loss_suite = LOSS_FF;
        break;
    case LOSS_SYMBA_TYPE:
        ffnet->loss_suite = LOSS_SYMBA;
        break;
    default:
        log_error("Unknown loss function type %d", ffnet->loss_suite.type);
        return;
    }

    for (int i = 0; i < ffnet->num_cells; i++)
        ffnet->layers[i] = load_ff_cell(file, act, pdact, beta1, beta2);

    fclose(file);
    log_info("Loaded FFNet from file %s", filename);
}