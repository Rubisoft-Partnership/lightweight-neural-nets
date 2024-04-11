/**
 * @file ff-net.c
 * @brief This file contains the implementation of a FFNet for the FF algorithm.
 *
 * */

#include <ff-net/ff-net.h>

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#include <logging/logging.h>
#include <ff-cell/ff-cell.h>

// Buffer to store output activations.
#define H_BUFFER_SIZE 1024
double o_buffer[H_BUFFER_SIZE]; // outputs buffer for positive pass

/**
 * @brief Builds a FFNet by creating multiple FFCell objects.
 *
 * @param layer_sizes The array of layer sizes, including the number of input and output units.
 * @param num_layers The number of layers in the FFNet.
 * @param act The activation function for the FFNet.
 * @param pdact The derivative of the activation function for the FFNet.
 * @param treshold The threshold value for the FFNet.
 * @param loss_suite The loss function suite for the FFNet.
 * @return FFNet The constructed FFNet.
 */
FFNet new_ff_net(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double treshold, Loss loss_suite)
{
    FFNet ffnet;
    ffnet.loss_suite = loss_suite;
    ffnet.num_cells = num_layers - 1;

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
        ffnet.layers[i] = new_ff_cell(layer_sizes[i], layer_sizes[i + 1], act, pdact, treshold);
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
 * @param rate The learning rate.
 * @return double The total loss of the FFNet.
 */
double train_ff_net(const FFNet ffnet, const double *const pos, const double *const neg, double rate)
{
    double loss = 0.0;
    loss += train_ff_cell(ffnet.layers[0], pos, neg, rate, ffnet.loss_suite);
    for (int i = 1; i < ffnet.num_cells; i++)
        loss += train_ff_cell(ffnet.layers[i], o_buffer, ffnet.layers[i - 1].output, rate, ffnet.loss_suite);
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
            fprop(ffnet.layers[i], i == 0 ? netinput : ffnet.layers[i - 1].output);
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
