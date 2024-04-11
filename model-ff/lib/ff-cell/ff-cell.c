/**
 * @file ff-cell.c
 * @brief This file contains the implementation of a feedforward neural network block.
 *
 * */

#include <ff-cell/ff-cell.h>

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#include <logging/logging.h>

// Buffer to store hidden activations and output activations.
extern double h_buffer[H_BUFFER_SIZE]; // activations buffer
extern double o_buffer[H_BUFFER_SIZE]; // outputs buffer

// Function declarations.
static void ffbprop(const Tinn t, const double *const in_pos, const double *const in_neg,
                    const double rate, const double g_pos, const double g_neg);
static double fferr(const double g_pos, const double g_neg, const double threshold);
static double ffpderr(const double g_pos, const double g_neg, const double threshold);
double goodness(const double *vec, const int size);
static double stable_sigmoid(double x);

// From Tinn.c, but modified
void fprop(const Tinn t, const double *const in);

// From Tinn.c
static void wbrand(const Tinn t);
static double frand(void);

// Generates inputs for inference given input and label
void embed_label(double *sample, const double *in, const int label, const int insize, const int num_classes)
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
        {
            const double pos_adjust = h_buffer[i] > 0.0 ? pderr_pos * 2 * sum_pos * in_pos[j] : 0.0;
            const double neg_adjust = h_buffer[i] > 0.0 ? pderr_neg * 2 * sum_neg * in_neg[j] : 0.0;
            t.w[i * t.nips + j] -= rate * (pos_adjust + neg_adjust);
        }
    }
}

// Computes error using the FFLoss function.
static double fferr(const double g_pos, const double g_neg, const double threshold)
{
    // double pos_exponent = -g_pos + threshold;
    // double neg_exponent = g_neg - threshold;

    // numerical stability fix:
    // double first_term = log(1 + exp(-fabs(pos_exponent))) + pos_exponent > 0.0 ? pos_exponent : 0.0;
    // double second_term = log(1 + exp(-fabs(neg_exponent))) + neg_exponent > 0.0 ? neg_exponent : 0.0;
    // log_debug("g_pos: %f, g_neg: %f, err: %f", g_pos, g_neg, first_term + second_term);
    // return first_term + second_term;
    // equivalent to:
    return log(1.0 + exp(-g_pos + threshold)) + log(1.0 + exp(g_neg - threshold));
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
void free_ff_cell(const Tinn t)
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
