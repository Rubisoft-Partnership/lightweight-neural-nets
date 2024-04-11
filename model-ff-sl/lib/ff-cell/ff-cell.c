/**
 * @file ff-cell.c
 * @brief This file contains the implementation of a FF cell for the FF algorithm.
 *
 * The FF cell is a building block of the FF algorithm, which is used for training neural networks.
 * This file contains the implementation of the FF cell, including functions for forward propagation,
 * backward propagation, weight initialization, and activation functions.
 */

#include <ff-cell/ff-cell.h>

#include <stdlib.h>
#include <math.h>

#include <logging/logging.h>
#include <utils/utils.h>
#include <losses/losses.h>
#include <ff-utils/ff-utils.h>

// Buffer to store activations and output activations for the positive pass.
extern double o_buffer[H_BUFFER_SIZE]; // outputs buffer

// Forward pass for a FF cell.
void fprop(const FFCell t, const double *const in);
// Backward pass for a FF cell.
static void bprop(const FFCell t, const double *const in_pos, const double *const in_neg,
                  const double rate, const double g_pos, const double g_neg, const Loss loss_suite);

// Random number generation for weights.
static void wbrand(FFCell t);
static double frand(void);

// Constructs a FF cell with number of inputs, number of outputs, activation function, and threshold.
FFCell new_ff_cell(const int input_size, const int output_size, double (*act)(double), double (*pdact)(double), const double threshold)
{
    FFCell t;

    // Adam optimizer
    t.adam = adam_create(0.9, 0.999, input_size * output_size);
    /// TODO: Fix hardcoding of Adam hyperparameters

    t.nw = input_size * output_size;                             // total number of weights
    t.weights = (double *)calloc(t.nw, sizeof(*t.weights));      // weights
    t.output = (double *)calloc(output_size, sizeof(*t.output)); // output neurons
    t.input_size = input_size;
    t.output_size = output_size;
    t.act = act;
    t.pdact = pdact;
    t.threshold = threshold;
    // Randomize weights and bias.
    wbrand(t);
    // Log the construction of the FF cell.
    increase_indent();
    log_debug("FFCell built with %d inputs, %d outputs, and %d weights", input_size, output_size, t.nw);
    decrease_indent();
    return t;
}

// Frees object from heap.
void free_ff_cell(const FFCell t)
{
    free(t.weights);
    free(t.output);
    adam_free(t.adam);
}

// Trains a FF cell performing forward and backward pass with given a loss function.
double train_ff_cell(const FFCell t, const double *const pos, const double *const neg, double rate, const Loss loss_suite)
{
    // Increase the indent level for logging
    increase_indent();

    // Positive forward pass.
    fprop(t, pos);
    // Copy positive activation output.
    memcpy(o_buffer, t.output, t.output_size * sizeof(*t.output));
    // Calculate the goodness of the positive pass.
    double g_pos = goodness(t.output, t.output_size);

    // Negative forward pass.
    fprop(t, neg);
    // Calculate the goodness of the negative pass.
    double g_neg = goodness(t.output, t.output_size);

    // Peforms weight update.
    bprop(t, pos, neg, rate, g_pos, g_neg, loss_suite);

    // Normalize the output of the layer
    normalize_vector(t.output, t.output_size);
    normalize_vector(o_buffer, t.output_size);

    // Calculate the average and standard deviation of weight values for debugging.
    double sum_weights = 0.0;
    double sum_weights_squared = 0.0;
    for (int i = 0; i < t.nw; i++)
    {
        sum_weights += t.weights[i];
        sum_weights_squared += t.weights[i] * t.weights[i];
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
void fprop(const FFCell t, const double *const in)
{
    double debug_sum = 0.0;
    log_debug("Computing forward propagation for FFCell with %d inputs and %d outputs", t.input_size, t.output_size);
    // Calculate the activation output for each output unit
    for (int i = 0; i < t.output_size; i++)
    {
        double sum = 0.0;
        // Calculate the weighted sum of the inputs
        for (int j = 0; j < t.input_size; j++)
            sum += in[j] * t.weights[i * t.input_size + j];
        // Store the output of the activation function
        t.output[i] = t.act(sum + t.bias);
        debug_sum += t.output[i]; // for debugging
    }
    log_debug("Overall activation output: %f", debug_sum);
}

// Performs backward pass for the FF algorithm.
static void bprop(const FFCell t, const double *const in_pos, const double *const in_neg,
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
    for (int i = 0; i < t.input_size; i++)
    {
        for (int j = 0; j < t.output_size; j++)
        {
            int wheight_index = j * t.input_size + i;

            // Calculate the gradient of the loss with respect to the weight for the positive and negative pass
            const double gradient_pos = pdloss_pos * 2.0 * o_buffer[j] * in_pos[i];
            const double gradient_neg = pdloss_neg * 2.0 * t.output[j] * in_neg[i];
            const double gradient = gradient_pos + gradient_neg;
            // log_debug("Positive correction gradient_pos: %.10g, negative correction gradient_neg: %.10g", gradient_pos, gradient_neg);

            // Weight update using Adam optimizer
            const double weight_update = rate * adam_weight_update(t.adam, gradient, wheight_index);

            // Update the weight
            t.weights[wheight_index] -= weight_update;
            // log_debug("Weight update: %.17g", weight_update);
            // log_debug("Weight after correction: %.17g", t.weights[j * t.input_size + i]);

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

// Randomizes tinn weights and bias.
static void wbrand(FFCell t)
{
    for (int i = 0; i < t.nw; i++)
        t.weights[i] = frand() - 0.5;
    t.bias = frand() - 0.5;
}

// Returns random doulbe in [0.0 - 1.0]
static double frand(void)
{
    return get_random() / (double)RAND_MAX;
}
