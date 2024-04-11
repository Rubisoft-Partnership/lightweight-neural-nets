/**
 * @file ff-cell.c
 * @brief This file contains the implementation of a FF cell for the FF algorithm.
 *
 * */

#include <ff-cell/ff-cell.h>

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#include <logging/logging.h>
#include <utils/utils.h>
#include <losses/losses.h>

// Buffer to store activations and output activations for the positive pass.
extern double o_buffer[H_BUFFER_SIZE]; // outputs buffer

// Forward pass for a FF cell.
void fprop(const Tinn t, const double *const in);
// Backward pass for a FF cell.
static void bprop(const Tinn t, const double *const in_pos, const double *const in_neg,
                  const double rate, const double g_pos, const double g_neg, const Loss loss_suite);

// Random number generation for weights.
static void wbrand(Tinn t);
static double frand(void);

// Constructs a FF cell with number of inputs, number of outputs, activation function, and threshold.
Tinn new_ff_cell(const int nips, const int nops, double (*act)(double), double (*pdact)(double), const double threshold)
{
    Tinn t;

    // Adam optimizer
    t.adam = adam_create(0.9, 0.999, nips * nops);
    /// TODO: Fix hardcoding of Adam hyperparameters

    t.nw = nips * nops;                         // total number of weights
    t.w = (double *)calloc(t.nw, sizeof(*t.w)); // weights (both [intput to hidden] and [hidden to output])
    t.o = (double *)calloc(nops, sizeof(*t.o)); // output neurons
    t.nips = nips;
    t.nops = nops;
    t.act = act;
    t.pdact = pdact;
    t.threshold = threshold;
    wbrand(t);
    increase_indent();
    log_debug("Tinn built with %d inputs, %d outputs, and %d weights", nips, nops, t.nw);
    decrease_indent();
    return t;
}

// Trains a FF cell performing forward and backward pass with given a loss function.
double train_ff_cell(const Tinn t, const double *const pos, const double *const neg, double rate, const Loss loss_suite)
{
    // Increase the indent level for logging
    increase_indent();

    // Positive forward pass.
    fprop(t, pos);
    // Copy positive activation output.
    memcpy(o_buffer, t.o, t.nops * sizeof(*t.o));
    // Calculate the goodness of the positive pass.
    double g_pos = goodness(t.o, t.nops);

    // Negative forward pass.
    fprop(t, neg);
    // Calculate the goodness of the negative pass.
    double g_neg = goodness(t.o, t.nops);

    // Peforms weight update.
    bprop(t, pos, neg, rate, g_pos, g_neg, loss_suite);

    // Normalize the output of the layer
    normalize_vector(t.o, t.nops);
    normalize_vector(o_buffer, t.nops);

    // Calculate the average and standard deviation of weight values for debugging.
    double sum_weights = 0.0;
    double sum_weights_squared = 0.0;
    for (int i = 0; i < t.nw; i++)
    {
        sum_weights += t.w[i];
        sum_weights_squared += t.w[i] * t.w[i];
    }
    double mean_weights = sum_weights / t.nw;
    double std_weights = sqrt((sum_weights_squared / t.nw) - (mean_weights * mean_weights));
    decrease_indent();
    log_info("Mean weight value: %f\n", mean_weights);
    log_info("Standard deviation of weight value: %f\n", std_weights);

    // Return the loss of the layer
    return loss_suite.loss(g_pos, g_neg, t.threshold);
}

// Performs forward propagation.
void fprop(const Tinn t, const double *const in)
{
    double debug_sum = 0.0;
    log_debug("Computing forward propagation for Tinn with %d inputs and %d outputs", t.nips, t.nops);
    // Calculate the activation output for each output unit
    for (int i = 0; i < t.nops; i++)
    {
        double sum = 0.0;
        // Calculate the weighted sum of the inputs
        for (int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        // Store the output of the activation function
        t.o[i] = t.act(sum + t.b);
        debug_sum += t.o[i]; // for debugging
    }
    log_debug("Overall activation output: %f", debug_sum);
}

// Performs backward pass for the FF algorithm.
static void bprop(const Tinn t, const double *const in_pos, const double *const in_neg,
                  const double rate, const double g_pos, const double g_neg, const Loss loss_suite)
{
    // Calculate the partial derivative of the loss with respect to the goodness of the positive and negative pass.
    const double pdloss_pos = loss_suite.pdloss_pos(g_pos, g_neg, t.threshold);
    const double pdloss_neg = loss_suite.pdloss_neg(g_pos, g_neg, t.threshold);
    log_debug("G_pos: %f, G_neg: %f", g_pos, g_neg);
    log_debug("Loss: %.17g", loss_suite.loss(g_pos, g_neg, t.threshold));
    log_debug("Partial derivative of the loss with resect to the goodness pos: %.17g, neg: %.17g", pdloss_pos, pdloss_neg);

    // Debugging variables statistics about weight updates.
    int updated_weights = 0;
    double sum_weight_update = 0.0;
    double sum_weight_update_squared = 0.0;

    // Update the weights for each connection between input and output units
    for (int i = 0; i < t.nips; i++)
    {
        for (int j = 0; j < t.nops; j++)
        {
            int wheight_index = j * t.nips + i;

            // Calculate the gradient of the loss with respect to the weight for the positive and negative pass
            const double gradient_pos = pdloss_pos * 2.0 * o_buffer[j] * in_pos[i];
            const double gradient_neg = pdloss_neg * 2.0 * t.o[j] * in_neg[i];
            const double gradient = gradient_pos + gradient_neg;
            // log_debug("Positive correction gradient_pos: %.10g, negative correction gradient_neg: %.10g", gradient_pos, gradient_neg);

            // Weight update using Adam optimizer
            const double weight_update = rate * adam_weight_update(t.adam, gradient, wheight_index);

            // Update the weight
            t.w[wheight_index] -= weight_update;
            // log_debug("Weight update: %.17g", weight_update);
            // log_debug("Weight after correction: %.17g", t.w[j * t.nips + i]);

            // Update statistics about weight updates.
            if (weight_update != 0.0)
            {
                updated_weights++;
                sum_weight_update += weight_update;
                sum_weight_update_squared += weight_update * weight_update;
            }
        }
    }

    // Log statistics about weight updates.
    double mean_weight_update = 0.0;
    double std_weight_update = 0.0;
    if (updated_weights != 0)
    {
        mean_weight_update = sum_weight_update / updated_weights;
        std_weight_update = sqrt((sum_weight_update_squared / updated_weights) - (mean_weight_update * mean_weight_update));
    }
    log_debug("Updated weights: %d\n", updated_weights);
    log_debug("Mean weight update: %f\n", mean_weight_update);
    log_debug("Standard deviation of weight update: %f\n", std_weight_update);
}

// Normalizes a vector.
void normalize_vector(double *vec, int size)
{
    double norm = 0.0;
    for (int i = 0; i < size; i++)
        norm += vec[i] * vec[i];
    norm = sqrt(norm);
    for (int i = 0; i < size; i++)
        vec[i] /= norm;
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
    return a > 0.0 ? a : 0.0;
}

// ReLU derivative.
double pdrelu(const double a)
{
    return a > 0.0 ? 1.0 : 0.0;
}

// Generates inputs for inference given input and label
void embed_label(double *sample, const double *in, const int label, const int in_size, const int num_classes)
{
    memcpy(sample, in, in_size * sizeof(*in));
    memset(&sample[in_size - num_classes], 0, num_classes * sizeof(*sample));
    sample[in_size - num_classes + label] = 1.0;
}

// Frees object from heap.
void free_ff_cell(const Tinn t)
{
    free(t.w);
    free(t.o);
    adam_free(t.adam);
}

// Randomizes tinn weights and biases.
static void wbrand(Tinn t)
{
    for (int i = 0; i < t.nw; i++)
        t.w[i] = frand() - 0.5;
    t.b = frand() - 0.5;
}

// Returns random doulbe in [0.0 - 1.0]
static double frand(void)
{
    return get_random() / (double)RAND_MAX;
}
