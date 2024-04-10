/**
 * @file ff-net.c
 * @brief This file contains the implementation of a feedforward neural network.
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

// Buffer to store hidden activations and output activations.
#define H_BUFFER_SIZE 1024
double h_buffer[H_BUFFER_SIZE]; // activations buffer
double o_buffer[H_BUFFER_SIZE]; // outputs buffer


// Builds a FFNet by creating multiple Tinn objects. layer_sizes includes the number of inputs, hidden neurons, and outputs units.
FFNet ffnetbuild(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double treshold)
{
    FFNet ffnet;
    ffnet.num_layers = num_layers;
    ffnet.num_hid_layers = num_layers - 2;

    // begin logs
    log_info("Building FFNet with %d layers, %d hidden layers", num_layers, ffnet.num_hid_layers);
    // logs layers dimensions in a single line
    char layers_str[256];
    layers_str[0] = '\0';
    for (int i = 0; i < num_layers; i++)
    {
        char layer_str[32];
        snprintf(layer_str, sizeof(layer_str), "%d ", layer_sizes[i]);
        strcat(layers_str, layer_str);
    }
    log_info("Layers: %s", layers_str);
    // end logs

    for (int i = 1; i < num_layers - 1; i++)
    {
        ffnet.hid_layers[i - 1] = new_ff_cell(layer_sizes[i - 1], layer_sizes[i], layer_sizes[i + 1], act, pdact, treshold);
    }

    return ffnet;
}

double fftrainnet(const FFNet ffnet, const double *const pos, const double *const neg, double rate)
{
    // printf("Training FFNet...\n");
    double error = 0.0;
    // Feed first layer manually.
    error += fftrain(ffnet.hid_layers[0], pos, neg, rate);
    // Feed the rest of the layers.
    for (int i = 1; i < ffnet.num_hid_layers; i++)
    {
        error += fftrain(ffnet.hid_layers[i], o_buffer, ffnet.hid_layers[i - 1].o, rate);
    }
    // printf("error: %f\n", error);
    return error;
}

// Inference function for FFNet.
int ffpredictnet(const FFNet ffnet, const double *in, const int num_classes, const int insize)
{
    double *netinput = (double *)malloc((insize) * sizeof(double));
    double goodnesses[MAX_CLASSES];
    for (int label = 0; label < num_classes; label++)
    {
        embed_label(netinput, in, label, insize, num_classes);
        fprop(ffnet.hid_layers[0], in);
        normalize_vector(ffnet.hid_layers[0].o, ffnet.hid_layers[0].nops);
        for (int i = 1; i < ffnet.num_hid_layers; i++)
        {
            fprop(ffnet.hid_layers[i], ffnet.hid_layers[i - 1].o);
            normalize_vector(ffnet.hid_layers[i].o, ffnet.hid_layers[i].nops);
            goodnesses[label] += goodness(ffnet.hid_layers[i].o, ffnet.hid_layers[i].nops);
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
