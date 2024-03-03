/**
 * @file ff-lib.c
 * @brief This file contains the implementation of the feedforward neural network library.
 *
 * */

#include "ff-lib.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#include "logging.h"

// Buffer to store hidden activations and output activations.
#define H_BUFFER_SIZE 1024
double h_buffer[H_BUFFER_SIZE]; // activations buffer
double o_buffer[H_BUFFER_SIZE]; // outputs buffer

#define MAX_CLASSES 16

// Function declarations.
static void ffbprop(const Tinn t, const double *const in_pos, const double *const in_neg,
                    const double rate, const double g_pos, const double g_neg);
static double fferr(const double g_pos, const double g_neg, const double threshold);
static double ffpderr(const double g_pos, const double g_neg, const double threshold);
static double goodness(const double *vec, const int size);
double fftrain(const Tinn t, const double *const pos, const double *const neg, double rate);
Tinn xtbuild(const int nips, const int nhid, const int nops, double (*act)(double), double (*pdact)(double), const double threshold);
void embed_label(double *sample, const double *in, int label, int insize, int num_classes);
void normalize_vector(double *output, int size);
static double stable_sigmoid(double x);

// From Tinn.c, but modified
void fprop(const Tinn t, const double *const in);

// From Tinn.c
static void wbrand(const Tinn t);
static double frand(void);


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
        ffnet.hid_layers[i - 1] = xtbuild(layer_sizes[i - 1], layer_sizes[i], layer_sizes[i + 1], act, pdact, treshold);
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
int ffpredictnet(const FFNet ffnet, const double *in, int num_classes, int insize)
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

// Generates inputs for inference given input and label
void embed_label(double *sample, const double *in, int label, int insize, int num_classes)
{
    memcpy(sample, in, insize * sizeof(*in));
    memset(&sample[insize - num_classes], 0, num_classes * sizeof(*sample));
    sample[insize - label] = 1.0;
}

// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
double fftrain(const Tinn t, const double *const pos, const double *const neg, double rate)
{
    // Positive pass.
    fprop(t, pos);
    memcpy(h_buffer, t.h, t.nhid * sizeof(*t.h)); // copy activation and output
    memcpy(o_buffer, t.o, t.nops * sizeof(*t.o));
    double g_pos = goodness(t.o, t.nops);

    // Negative pass.
    fprop(t, neg);
    double g_neg = goodness(t.o, t.nops);

    // Peforms gradient descent.
    ffbprop(t, pos, neg, rate, g_pos, g_neg);

    // Normalize the output of the layer
    normalize_vector(t.o, t.nops);
    normalize_vector(o_buffer, t.nops);

    // printf("g_pos: %f, g_neg: %f, err: %f\n", g_pos, g_neg, fferr(g_pos, g_neg, t.threshold));
    return fferr(g_pos, g_neg, t.threshold);
}

void normalize_vector(double *output, int size)
{
    double norm = 0.0;
    for (int i = 0; i < size; i++)
        norm += output[i] * output[i];
    norm = sqrt(norm);
    for (int i = 0; i < size; i++)
        output[i] /= norm;
}

// Performs back propagation for the FF algorithm.
static void ffbprop(const Tinn t, const double *const in_pos, const double *const in_neg,
                    const double rate, const double g_pos, const double g_neg)
{
    // const double a = ffpderr(g_pos, g_neg, t.threshold);
    const double pderr_pos = -stable_sigmoid(t.threshold - g_pos);
    const double pderr_neg = stable_sigmoid(g_neg - t.threshold);
    log_debug("G_pos: %f, G_neg: %f", g_pos, g_neg);
    log_debug("Loss: %.17g", fferr(g_pos, g_neg, t.threshold));
    log_debug("Partial derivative pos: %.17g, neg: %.17g", pderr_pos, pderr_neg);

    for (int i = 0; i < t.nhid; i++)
    {
        double sum_pos = 0.0;
        double sum_neg = 0.0;
        // Calculate total error change with respect to output.
        for (int j = 0; j < t.nops; j++)
        {
            // Computes the positive and negative:
            // derivative of the goodness with respect to the output of the layer
            // multiplied by the derivative of the output of the layer with respect to the weights in the hidden layer to output layer
            const double b_pos = 2 * o_buffer[j] * h_buffer[i];
            const double b_neg = 2 * t.o[j] * t.h[i];
            sum_pos += o_buffer[j] * t.x[j * t.nhid + i];
            sum_neg += t.o[j] * t.x[j * t.nhid + i];
            // Correct weights in hidden to output layer.
            t.x[j * t.nhid + i] -= rate * (b_pos * pderr_pos + b_neg * pderr_neg);
        }
        // Correct weights in input to hidden layer.
        for (int j = 0; j < t.nips; j++)
            t.w[i * t.nips + j] -= rate * ((pderr_neg * 2 * sum_neg * in_neg[j]) + (pderr_pos * 2 * sum_pos * in_pos[j]));
    }
}

// Computes error using the FFLoss function.
static double fferr(const double g_pos, const double g_neg, const double threshold)
{
    double pos_exponent = -g_pos + threshold;
    double neg_exponent = g_neg - threshold;
    // double first_term = log(1 + exp(-fabs(pos_exponent))) + pos_exponent > 0.0 ? pos_exponent : 0.0;
    // double second_term = log(1 + exp(-fabs(neg_exponent))) + neg_exponent > 0.0 ? neg_exponent : 0.0;
    // printf("g_pos: %f, g_neg: %f, err: %f\n", g_pos, g_neg, first_term + second_term);
    // return first_term + second_term;
    // equivalent to:
    return logf(1.0 + expf(-g_pos + threshold)) + logf(1.0 + expf(g_neg - threshold));
}

static double stable_sigmoid(double x)
{
    if (x >= 0)
    {
        return 1.0 / (1.0 + exp(-x) + 1e-4);
    }
    else
    {
        /// TODO: Fix underflow here
        double exp_x = exp(x);
        log_debug("x: %.17g, exp_x: %.17g", x, exp_x);
        return exp_x / (1.0 + exp_x + 1e-4);
    }
}

// Returns partial derivative of error function.
static double ffpderr(const double g_pos, const double g_neg, const double threshold)
{
    double sigmoid_g_pos = stable_sigmoid(threshold - g_pos);
    double sigmoid_g_neg = stable_sigmoid(g_neg - threshold);

    log_debug("Sigmoid g_pos: %.17g, Sigmoid g_neg: %.17g", sigmoid_g_pos, sigmoid_g_neg);
    return sigmoid_g_neg - sigmoid_g_pos;
    // Equivalent to:
    // return -expf(threshold) / (expf(g_pos) + expf(threshold)) + expf(threshold) / (expf(g_neg) + expf(threshold));
}

// Returns the goodness of a layer.
double goodness(const double *vec, const int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        sum += vec[i] * vec[i];
    }
    return sum;
}

// ReLU activation function.
double relu(const double a)
{
    // return a;
    return a > 0.0 ? a : 0.0;
}

// ReLU derivative.
double pdrelu(const double a)
{
    // return 1;
    return a > 0.0 ? 1.0 : 0.0;
}

// Sigmoid activation function.
double sigmoid(const double a)
{
    return 1.0 / (1.0 + exp(-a));
}

// Sigmoid derivative.
double pdsigmoid(const double a)
{
    return a * (1.0 - a);
}

// Performs forward propagation.
void fprop(const Tinn t, const double *const in)
{
    // Calculate hidden layer neuron values.
    for (int i = 0; i < t.nhid; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.h[i] = t.act(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for (int i = 0; i < t.nops; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < t.nhid; j++)
            sum += t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = t.act(sum + t.b[1]);
    }
}

// Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
Tinn xtbuild(const int nips, const int nhid, const int nops, double (*act)(double), double (*pdact)(double), const double threshold)
{
    Tinn t;
    // Tinn only supports one hidden layer so there are two biases.
    t.nb = 2;
    t.nw = nhid * (nips + nops);                // total number of weights
    t.w = (double *)calloc(t.nw, sizeof(*t.w)); // weights (both [intput to hidden] and [hidden to output])
    t.x = t.w + nhid * nips;
    t.b = (double *)calloc(t.nb, sizeof(*t.b)); // biases
    t.h = (double *)calloc(nhid, sizeof(*t.h)); // hidden neurons
    t.o = (double *)calloc(nops, sizeof(*t.o)); // output neurons
    t.nips = nips;
    t.nhid = nhid;
    t.nops = nops;
    t.act = act;
    t.pdact = pdact;
    t.threshold = threshold;
    wbrand(t);
    return t;
}

/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Below is the original Tinn code. It is in this file because of the way the Tinn library is structured.
/// TODO: Fix imports and files structure.

// Loads a tinn from disk.
Tinn xtload(const char *const path)
{
    FILE *const file = fopen(path, "r");
    int nips = 0;
    int nhid = 0;
    int nops = 0;
    // Load header.
    fscanf(file, "%d %d %d\n", &nips, &nhid, &nops);
    // Build a new tinn.
    const Tinn t = xtbuild(nips, nhid, nops, sigmoid, pdsigmoid, 0.5); /// TODO: relu and treshold hardcode is a quick fix, change this!
    // Load bias and weights.
    for (int i = 0; i < t.nb; i++)
        fscanf(file, "%lf\n", &t.b[i]);
    for (int i = 0; i < t.nw; i++)
        fscanf(file, "%lf\n", &t.w[i]);
    fclose(file);
    return t;
}

// Saves a tinn to disk.
void xtsave(const Tinn t, const char *const path)
{
    FILE *const file = fopen(path, "w");
    // Save header.
    fprintf(file, "%d %d %d\n", t.nips, t.nhid, t.nops);
    // Save biases and weights.
    for (int i = 0; i < t.nb; i++)
        fprintf(file, "%f\n", (double)t.b[i]);
    for (int i = 0; i < t.nw; i++)
        fprintf(file, "%f\n", (double)t.w[i]);
    fclose(file);
}

// Frees object from heap.
void xtfree(const Tinn t)
{
    free(t.w);
    free(t.b);
    free(t.h);
    free(t.o);
}

// Prints an array of doubles. Useful for printing predictions.
void xtprint(const double *arr, const int size)
{
    for (int i = 0; i < size; i++)
        printf("%f ", (double)arr[i]);
    printf("\n");
}

// Returns an output prediction given an input.
double *xtpredict(const Tinn t, const double *const in)
{
    fprop(t, in);
    return t.o;
}

// Randomizes tinn weights and biases.
static void wbrand(const Tinn t)
{
    for (int i = 0; i < t.nw; i++)
        t.w[i] = frand() - 0.5f;
    for (int i = 0; i < t.nb; i++)
        t.b[i] = frand() - 0.5f;
}

// Returns doubleing point random from 0.0 - 1.0.
static double frand(void)
{
    return rand() / (double)RAND_MAX;
}
