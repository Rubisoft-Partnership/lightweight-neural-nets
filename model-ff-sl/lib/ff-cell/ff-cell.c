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

// Buffer to store activations and output activations for the positive pass.
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
static void wbrand(Tinn t);
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
    // Copy positive activation output.
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

    // Calculate the average and standard deviation of weight values
    double sum_weights = 0.0;
    double sum_weights_squared = 0.0;
    for (int i = 0; i < t.nw; i++) {
        sum_weights += t.w[i];
        sum_weights_squared += t.w[i] * t.w[i];
    }
    double mean_weights = sum_weights / t.nw;
    double std_weights = sqrt((sum_weights_squared / t.nw) - (mean_weights * mean_weights));
    log_info("Mean weight value: %f\n", mean_weights);
    log_info("Standard deviation of weight value: %f\n", std_weights);

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

    int updated_weights = 0;
    double sum_weight_update = 0.0;
    double sum_weight_update_squared = 0.0;

    for (int i = 0; i < t.nips; i++)
    {
        for (int j = 0; j < t.nops; j++)
        {   
            // log_debug("Weight from unit [%d] to unit [%d]: %.17g", i, j, t.w[j * t.nips + i]);
            const double b_pos = pderr_pos * 2.0 * o_buffer[j] * in_pos[i];
            const double b_neg = pderr_neg * 2.0 * t.o[j] * in_neg[i];
            // log_debug("Positive correction b_pos: %.10g, negative correction b_neg: %.10g", b_pos, b_neg);
            double weight_update = rate * (b_pos + b_neg);
            t.w[j * t.nips + i] -= weight_update;
            // log_debug("Weight update: %.17g", weight_update);
            // log_debug("Weight after correction: %.17g", t.w[j * t.nips + i]);

            if (weight_update != 0.0) {
                updated_weights++;
                sum_weight_update += weight_update;
                sum_weight_update_squared += weight_update * weight_update;
            }
        }
    }

    // Log statistics about weight updates.
    double mean_weight_update = 0.0;
    double std_weight_update = 0.0;
    if (updated_weights != 0) {
        mean_weight_update = sum_weight_update / updated_weights;
        std_weight_update = sqrt((sum_weight_update_squared / updated_weights) - (mean_weight_update * mean_weight_update));
    }
    log_debug("Updated weights: %d\n", updated_weights);
    log_debug("Mean weight update: %f\n", mean_weight_update);
    log_debug("Standard deviation of weight update: %f\n", std_weight_update);
}

// Computes error using the FFLoss function.
static double fferr(const double g_pos, const double g_neg, const double threshold)
{
    double pos_exponent = -g_pos + threshold;
    double neg_exponent = g_neg - threshold;

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
        return 1.0 / (1.0 + exp(-x) + 1e-4);
    else
    {
        /// TODO: Fix underflow here
        double exp_x = exp(x);
        // log_debug("sigmoid input x: %.17g, exp_x: %.17g", x, exp_x);
        return exp_x / (1.0 + exp_x + 1e-4);
    }
}

// Returns the goodness of a layer.
double goodness(const double *vec, const int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += vec[i] * vec[i];
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
    double debug_sum = 0.0;
    log_debug("Computing forward propagation for Tinn with %d inputs and %d outputs", t.nips, t.nops);
    // Calculate hidden layer neuron values.
    for (int i = 0; i < t.nops; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.o[i] = t.act(sum + t.b);
        debug_sum += t.o[i]; // for debugging
    }
    log_debug("Overall activation output: %f", debug_sum);
}

// Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
Tinn xtbuild(const int nips, const int nops, double (*act)(double), double (*pdact)(double), const double threshold)
{
    Tinn t;
    t.nw = nips * nops;                         // total number of weights
    t.w = (double *)calloc(t.nw, sizeof(*t.w)); // weights (both [intput to hidden] and [hidden to output])
    t.o = (double *)calloc(nops, sizeof(*t.o)); // output neurons
    t.nips = nips;
    t.nops = nops;
    t.act = act;
    t.pdact = pdact;
    t.threshold = threshold;
    wbrand(t);
    log_debug("Tinn built with %d inputs, %d outputs, and %d weights", nips, nops, t.nw);
    return t;
}

// Frees object from heap.
void xtfree(const Tinn t)
{
    free(t.w);
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
static void wbrand(Tinn t)
{
    for (int i = 0; i < t.nw; i++)
        t.w[i] = frand() - 0.5;
    t.b = frand() - 0.5;
}

// Returns doubleing point random from 0.0 - 1.0.
static double frand(void)
{
    return rand() / (double)RAND_MAX;
}
