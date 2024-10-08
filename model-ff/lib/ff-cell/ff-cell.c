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
#include <data/data.h>
#include <losses/losses.h>
#include <ff-utils/ff-utils.h>

/**
 * Performs the forward pass for a feedforward (FF) cell.
 *
 * @param ffcell The FFCell object representing the FF cell.
 * @param in The input data for the forward pass.
 */
void fprop_ff_cell(const FFCell ffcell, const double *const in);

/**
 * Performs the backward pass for a feedforward (FF) cell.
 *
 * @param ffcell The FF cell to perform the backward pass on.
 * @param learning_rate The learning rate for the cell.
 */
static void bprop(const FFCell ffcell, const double learning_rate);

/**
 * Computes the gradient for the given of a sample of the batch for the FF cell and stores it in the gradient array.
 *
 * @param ffcell The FFCell for which the gradient is computed.
 * @param in_pos The positive input values.
 * @param in_neg The negative input values.
 * @param positive_output_buffer The positive output buffer.
 * @param g_pos The positive goodness value.
 * @param g_neg The negative goodness value.
 * @param threshold The threshold value.
 * @param loss_suite The loss function suite to be used.
 */
static void compute_gradient(const FFCell ffcell, const double *const in_pos, const double *const in_neg,
                             const double *const positive_output_buffer, const double g_pos, const double g_neg,
                             const double threshold, const Loss loss_suite);

// Random number generation for weights.
static void wbrand(FFCell *ffcell);
static double frand(void);

/**
 * Constructs a FF cell with the specified number of inputs, number of outputs, activation function,
 * and threshold.
 *
 * @param input_size The number of inputs for the FF cell.
 * @param output_size The number of outputs for the FF cell.
 * @param act The activation function for the FF cell.
 * @param pdact The derivative of the activation function for the FF cell.
 * @param beta1 The beta1 parameter for the Adam optimizer.
 * @param beta2 The beta2 parameter for the Adam optimizer.
 * @return The constructed FF cell.
 */
FFCell new_ff_cell(const int input_size, const int output_size, double (*act)(double),
                   double (*pdact)(double), const double beta1, const double beta2)
{
    FFCell ffcell;
    ffcell.num_weights = input_size * output_size; // total number of weights

    // Adam optimizer
    ffcell.adam = adam_create(beta1, beta2, ffcell.num_weights);

    ffcell.weights = (double *)calloc(ffcell.num_weights, sizeof(*ffcell.weights));   // weights
    ffcell.output = (double *)calloc(output_size, sizeof(*ffcell.output));            // output neurons
    ffcell.gradient = (double *)calloc(ffcell.num_weights, sizeof(*ffcell.gradient)); // gradient of each weight
    ffcell.input_size = input_size;
    ffcell.output_size = output_size;
    ffcell.act = act;
    ffcell.pdact = pdact;
    // Randomize weights and bias.
    wbrand(&ffcell);
    // Log the construction of the FF cell.
    increase_indent();
    log_debug("FFCell built with %d inputs, %d outputs, and %d weights", input_size, output_size, ffcell.num_weights);
    decrease_indent();
    return ffcell;
}

// Frees object from heap.
void free_ff_cell(const FFCell ffcell)
{
    free(ffcell.weights);
    free(ffcell.output);
    free(ffcell.gradient);
    adam_free(ffcell.adam);
}

/**
 * @brief Trains a FFCell by performing forward and backward pass with a given a batch of data.
 * @param ffcell The FFCell to be trained.
 * @param batch The batch of data to train on.
 * @param learning_rate The learning rate for the training.
 * @param threshold The threshold value for the FFCell.
 * @param loss_suite The loss function suite.
 * @return The loss value after training.
 */
double train_ff_cell(const FFCell ffcell, FFBatch batch, const double learning_rate, const double threshold, const LossType loss)
{
    Loss loss_suite = select_loss(loss);

    // Increase the indent level for logging
    increase_indent();
    double g_pos = 0.0, g_neg = 0.0;
    double loss_value = 0.0;

    double *positive_output_buffer = malloc(ffcell.output_size * sizeof(*positive_output_buffer));

    for (int i = 0; i < batch.size; i++)
    {
        // Get the positive and negative samples from the batch
        double *pos = batch.pos[i];
        double *neg = batch.neg[i];

        // Positive forward pass.
        fprop_ff_cell(ffcell, pos);
        // Copy positive activation output.
        memcpy(positive_output_buffer, ffcell.output, ffcell.output_size * sizeof(*ffcell.output));
        // Calculate the goodness of the positive pass.
        g_pos = goodness(ffcell.output, ffcell.output_size);

        // Negative forward pass.
        fprop_ff_cell(ffcell, neg);
        // Calculate the goodness of the negative pass.
        g_neg = goodness(ffcell.output, ffcell.output_size);

        // Compute and accumulate the gradient of the loss function with respect to the weights.
        compute_gradient(ffcell, pos, neg, positive_output_buffer, g_pos, g_neg, threshold, loss_suite);

        // Copy the positive and negative activation output for normalization.
        memcpy(pos, positive_output_buffer, ffcell.output_size * sizeof(*positive_output_buffer));
        memcpy(neg, ffcell.output, ffcell.output_size * sizeof(*ffcell.output));

        // Normalize the output in order to feed it to the next layer.
        normalize_vector(pos, ffcell.output_size);
        normalize_vector(neg, ffcell.output_size);

        loss_value += loss_suite.loss(g_pos, g_neg, threshold);
    }

    // Compute mean gradient of the batch.
    for (int i = 0; i < ffcell.num_weights; i++)
        ffcell.gradient[i] /= batch.size;
    // Performs weight update.
    bprop(ffcell, learning_rate);

    // Calculate the average and standard deviation of weight values for debugging.
    double sum_weights = 0.0;
    double sum_weights_squared = 0.0;
    for (int i = 0; i < ffcell.num_weights; i++)
    {
        sum_weights += ffcell.weights[i];
        sum_weights_squared += ffcell.weights[i] * ffcell.weights[i];
    }
    double mean_weights = sum_weights / ffcell.num_weights;
    double std_weights = sqrt((sum_weights_squared / ffcell.num_weights) - (mean_weights * mean_weights));
    decrease_indent();
    log_info("Mean weight value: %f\n", mean_weights);
    log_info("Standard deviation of weight value: %f\n", std_weights);

    // Free the positive output buffer.
    free(positive_output_buffer);

    // Return the loss of the layer
    return loss_value / batch.size;
}

// Performs forward propagation.
void fprop_ff_cell(const FFCell ffcell, const double *const in)
{
    double debug_sum = 0.0;
    log_debug("Computing forward propagation for FFCell with %d inputs and %d outputs", ffcell.input_size, ffcell.output_size);
    // Calculate the activation output for each output unit
    for (int i = 0; i < ffcell.output_size; i++)
    {
        double sum = 0.0;
        // Calculate the weighted sum of the inputs
        for (int j = 0; j < ffcell.input_size; j++)
            sum += in[j] * ffcell.weights[i * ffcell.input_size + j];
        // Store the output of the activation function
        ffcell.output[i] = ffcell.act(sum + ffcell.bias);
        debug_sum += ffcell.output[i]; // for debugging
    }
    log_debug("Overall activation output: %f", debug_sum);
}

static void compute_gradient(const FFCell ffcell, const double *const in_pos, const double *const in_neg,
                             const double *const positive_output_buffer, const double g_pos, const double g_neg,
                             const double threshold, const Loss loss_suite)
{
    log_debug("Computing gradient for FFCell with %d inputs and %d outputs", ffcell.input_size, ffcell.output_size);
    // Calculate the partial derivative of the loss with respect to the goodness of the positive and negative pass.
    const double pdloss_pos = loss_suite.pdloss_pos(g_pos, g_neg, threshold);
    const double pdloss_neg = loss_suite.pdloss_neg(g_pos, g_neg, threshold);
    log_debug("G_pos: %f, G_neg: %f", g_pos, g_neg);
    log_debug("Loss: %.17g", loss_suite.loss(g_pos, g_neg, threshold));
    log_debug("Partial derivative of the loss with resect to the goodness pos: %.17g, neg: %.17g", pdloss_pos, pdloss_neg);

    /// TODO: change loops to i < ffcell.num_weights
    for (int i = 0; i < ffcell.input_size; i++)
    {
        for (int j = 0; j < ffcell.output_size; j++)
        {
            int weight_index = j * ffcell.input_size + i;

            // Calculate the gradient of the loss with respect to the weight for the positive and negative pass
            double gradient_pos = pdloss_pos * 2.0 * positive_output_buffer[j] * in_pos[i];
            double gradient_neg = pdloss_neg * 2.0 * ffcell.output[j] * in_neg[i];
            ffcell.gradient[weight_index] += gradient_pos + gradient_neg; // accumulate the gradient
        }
    }
}

// Performs backward pass for the FF algorithm.
static void bprop(const FFCell ffcell, const double learning_rate)
{
    log_debug("Performing backward pass for FFCell with %d inputs and %d outputs", ffcell.input_size, ffcell.output_size);
    // Debugging variables statistics about weight updates.
    int updated_weights = 0;
    double sum_weight_update = 0.0;
    double sum_weight_update_squared = 0.0;

    // Update the weights for each connection between input and output units

    /// TODO: change loops to i < ffcell.num_weights
    for (int i = 0; i < ffcell.input_size; i++)
    {
        for (int j = 0; j < ffcell.output_size; j++)
        {
            int weight_index = j * ffcell.input_size + i;

            // Weight update using Adam optimizer
            const double weight_update = learning_rate * adam_weight_update(ffcell.adam, ffcell.gradient[weight_index], weight_index);

            // Update the weight
            ffcell.weights[weight_index] -= weight_update;
            // log_debug("Weight update: %.17g", weight_update);
            // log_debug("Weight after correction: %.17g", ffcell.weights[j * ffcell.input_size + i]);

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

/**
 * Saves the FFCell structure to a file.
 *
 * This function writes the weights, bias, number of weights, input size, and output size
 * of the FFCell structure to the specified file.
 *
 * @param ffcell The FFCell structure to be saved.
 * @param file The file pointer to write the FFCell structure to.
 */
void save_ff_cell(const FFCell ffcell, FILE *file)
{
    log_debug("Saving FFCell with %d inputs, %d outputs, and %d weights", ffcell.input_size, ffcell.output_size, ffcell.num_weights);
    // Write the input and output size of the FFCell to the file.
    fwrite(&ffcell.input_size, sizeof(ffcell.input_size), 1, file);
    fwrite(&ffcell.output_size, sizeof(ffcell.output_size), 1, file);

    // Save the weights and bias of the FFCell to the file.
    fwrite(ffcell.weights, sizeof(*ffcell.weights), ffcell.num_weights, file);
    fwrite(&ffcell.bias, sizeof(ffcell.bias), 1, file);
}

/**
 * Loads an FFCell object from a file.
 *
 * @param file The file to read the FFCell object from.
 * @param act Pointer to the activation function.
 * @param pdact Pointer to the derivative of the activation function.
 * @param beta1 The beta1 parameter for the FFCell object.
 * @param beta2 The beta2 parameter for the FFCell object.
 * @return The loaded FFCell object.
 */
FFCell load_ff_cell(FILE *file, double (*act)(double), double (*pdact)(double), const double beta1, const double beta2)
{
    // Read input and output size from file.
    int input_size = 0;
    int output_size = 0;
    size_t res;
    res = fread(&input_size, sizeof(input_size), 1, file);
    if (res != 1)
    {
        log_error("Failed to read input size from file");
        exit(1);
    }
    res = fread(&output_size, sizeof(output_size), 1, file);
    if (res != 1)
    {
        log_error("Failed to read output size from file");
        exit(1);
    }

    log_debug("Loading FFCell with %d inputs and %d outputs", input_size, output_size);

    // Allocate memory to create FFCell object.
    FFCell ffcell = new_ff_cell(input_size, output_size, act, pdact, beta1, beta2);

    // Load weights and bias from the file.
    res = fread(ffcell.weights, sizeof(*ffcell.weights), ffcell.num_weights, file);
    if (res != (size_t)ffcell.num_weights)
    {
        log_error("Failed to read weights from file");
        exit(1);
    }
    res = fread(&ffcell.bias, sizeof(ffcell.bias), 1, file);
    if (res != 1)
    {
        log_error("Failed to read bias from file");
        exit(1);
    }

    log_debug("FFCell loaded with %d inputs, %d outputs, and %d weights", ffcell.input_size, ffcell.output_size, ffcell.num_weights);
    return ffcell;
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

// Randomizes weights and bias.
static void wbrand(FFCell *ffcell)
{
    for (int i = 0; i < ffcell->num_weights; i++)
        ffcell->weights[i] = frand() - 0.5;
    ffcell->bias = frand() - 0.5;
}

// Returns random double in [0.0 - 1.0]
static double frand(void)
{
    return get_random() / (double)RAND_MAX;
}
