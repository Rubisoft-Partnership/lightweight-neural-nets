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


// Builds a FFNet by creating multiple Tinn objects. layer_sizes includes the number of inputs and outputs units.
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

// Frees the memory of a FFNet.
void ffnetfree(FFNet ffnet)
{
    for (int i = 0; i < ffnet.num_cells; i++)
        free_ff_cell(ffnet.layers[i]);
}

double fftrainnet(const FFNet ffnet, const double *const pos, const double *const neg, double rate)
{
    double error = 0.0;
    error += train_ff_cell(ffnet.layers[0], pos, neg, rate, ffnet.loss_suite);
    for (int i = 1; i < ffnet.num_cells; i++)
        error += train_ff_cell(ffnet.layers[i], o_buffer, ffnet.layers[i - 1].o, rate, ffnet.loss_suite);
    // printf("error: %f\n", error);
    return error;
}

// Inference function for FFNet.
int ffpredictnet(const FFNet ffnet, const double *in, const int num_classes, const int insize)
{
    log_debug("Predicting sample on model with cells: %d", ffnet.num_cells);
    double *netinput = (double *)malloc((insize) * sizeof(double));
    double goodnesses[MAX_CLASSES];
    // For debugging.
    for (int i = 0; i < num_classes; i++)
        goodnesses[i] = 0.0;

    for (int label = 0; label < num_classes; label++)
    {
        embed_label(netinput, in, label, insize, num_classes);
        for (int i = 0; i < ffnet.num_cells; i++)
        {
            fprop(ffnet.layers[i], i == 0 ? netinput : ffnet.layers[i - 1].o);
            goodnesses[label] += goodness(ffnet.layers[i].o, ffnet.layers[i].nops);
            normalize_vector(ffnet.layers[i].o, ffnet.layers[i].nops);
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
