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

#include <utils/utils.h>
#include <data/data.h>
#include <losses/losses.h>
#include <ff-utils/ff-utils.h>

// #include "buffers/buffers.h"

/**
 * Performs the forward pass for a feedforward (FF) cell.
 *
 * @param ffcell The FFCell object representing the FF cell.
 * @param in The input data for the forward pass.
 */
void fprop_ff_cell(const FFCell ffcell, const float *const in);

/**
 * Performs the backward pass for a feedforward (FF) cell.
 *
 * @param ffcell The FF cell to perform the backward pass on.
 * @param learning_rate The learning rate for the cell.
 */
static void bprop(FFCell ffcell, const float learning_rate);

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
static void compute_gradient(FFCell ffcell, const float *const in_pos, const float *const in_neg,
                             const float *const positive_output_buffer, const float g_pos, const float g_neg,
                             const float threshold, const Loss loss_suite);

// Random number generation for weights.
static void wbrand(FFCell *ffcell);
static float frand(void);

#define CHUNK_SIZE 1000
static void chunked_allocate(FFCell *ffcell)
{
    ffcell->weights = (float **)calloc(ffcell->num_weights / CHUNK_SIZE, sizeof(*ffcell->weights));
    if (ffcell->weights == NULL)
    {
        printf("Failed to allocate memory for weights\n");
    }
    int num_chunks = ffcell->num_weights / CHUNK_SIZE + 1;
    for (int i = 0; i < num_chunks; i++)
    {
        printf("Allocating chunk %d of %d\n", i, num_chunks);
        ffcell->weights[i] = (float *)calloc(CHUNK_SIZE, sizeof(*ffcell->weights[i]));
        if (ffcell->weights[i] == NULL)
        {
            printf("Failed to allocate memory for weights at chunk %d, chunk size %d\n", i, CHUNK_SIZE);
        }
    }
}

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
FFCell new_ff_cell(const int input_size, const int output_size, float (*act)(float),
                   float (*pdact)(float), const float beta1, const float beta2)
{
    FFCell ffcell;
    ffcell.num_weights = input_size * output_size; // total number of weights
    printf("ffcell.num_weights: %d\n", ffcell.num_weights);

    // Adam optimizer
    ffcell.adam = adam_create(beta1, beta2, ffcell.num_weights);

    // for (int i = 0; i < ffcell.num_weights; i++)
    // {
    //     ffcell.gradient[i] = 0.0;
    //     ffcell.weights[i] = 0.0;
    // }

    printf("Initializing FFCell...\n");

    // ffcell.weights = (float *)calloc(ffcell.num_weights, sizeof(*ffcell.weights));   // weights
    chunked_allocate(&ffcell);
    ffcell.output = (float *)calloc(output_size, sizeof(*ffcell.output));            // output neurons
    ffcell.gradient = (float *)calloc(ffcell.num_weights, sizeof(*ffcell.gradient)); // gradient of each weight
    ffcell.input_size = input_size;
    ffcell.output_size = output_size;
    ffcell.act = act;
    ffcell.pdact = pdact;
    // Randomize weights and bias.
    printf("Randomizing weights...\n");
    wbrand(&ffcell);
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
float train_ff_cell(FFCell ffcell, FFBatch batch, const float learning_rate, const float threshold, const LossType loss)
{
    Loss loss_suite = select_loss(loss);

    float g_pos = 0.0, g_neg = 0.0;
    float loss_value = 0.0;

    float *positive_output_buffer = malloc(ffcell.output_size * sizeof(*positive_output_buffer));

    for (int i = 0; i < batch.size; i++)
    {
        // Get the positive and negative samples from the batch
        float *pos = batch.pos[i];
        float *neg = batch.neg[i];

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

    // Free the positive output buffer.
    free(positive_output_buffer);

    // Return the loss of the layer
    return loss_value / batch.size;
}

// Performs forward propagation.
void fprop_ff_cell(const FFCell ffcell, const float *const in)
{
    float debug_sum = 0.0;
    // Calculate the activation output for each output unit
    for (int i = 0; i < ffcell.output_size; i++)
    {
        float sum = 0.0;
        // Calculate the weighted sum of the inputs
        for (int j = 0; j < ffcell.input_size; j++)
        {
            const int weight_index = i * ffcell.input_size + j;
            sum += in[j] * ffcell.weights[weight_index / CHUNK_SIZE][weight_index % CHUNK_SIZE];
        }
        // Store the output of the activation function
        ffcell.output[i] = ffcell.act(sum + ffcell.bias);
        debug_sum += ffcell.output[i]; // for debugging
    }
}

static void compute_gradient(FFCell ffcell, const float *const in_pos, const float *const in_neg,
                             const float *const positive_output_buffer, const float g_pos, const float g_neg,
                             const float threshold, const Loss loss_suite)
{
    // Calculate the partial derivative of the loss with respect to the goodness of the positive and negative pass.
    const float pdloss_pos = loss_suite.pdloss_pos(g_pos, g_neg, threshold);
    const float pdloss_neg = loss_suite.pdloss_neg(g_pos, g_neg, threshold);

    /// TODO: change loops to i < ffcell.num_weights
    for (int i = 0; i < ffcell.input_size; i++)
    {
        for (int j = 0; j < ffcell.output_size; j++)
        {
            int weight_index = j * ffcell.input_size + i;

            // Calculate the gradient of the loss with respect to the weight for the positive and negative pass
            float gradient_pos = pdloss_pos * 2.0 * positive_output_buffer[j] * in_pos[i];
            float gradient_neg = pdloss_neg * 2.0 * ffcell.output[j] * in_neg[i];
            ffcell.gradient[weight_index] += gradient_pos + gradient_neg; // accumulate the gradient
        }
    }
}

// Performs backward pass for the FF algorithm.
static void bprop(FFCell ffcell, const float learning_rate)
{

    // Update the weights for each connection between input and output units

    /// TODO: change loops to i < ffcell.num_weights
    for (int i = 0; i < ffcell.input_size; i++)
    {
        for (int j = 0; j < ffcell.output_size; j++)
        {
            int weight_index = j * ffcell.input_size + i;

            // Weight update using Adam optimizer
            const float weight_update = learning_rate * adam_weight_update(ffcell.adam, ffcell.gradient[weight_index], weight_index);

            // Update the weight
            ffcell.weights[weight_index / CHUNK_SIZE][weight_index % CHUNK_SIZE] -= weight_update;
        }
    }
}

// ReLU activation function.
float relu(const float a)
{
    return a > 0.0 ? a : 0.0;
}

// ReLU derivative.
float pdrelu(const float a)
{
    return a > 0.0 ? 1.0 : 0.0;
}

// Randomizes weights and bias.
static void wbrand(FFCell *ffcell)
{
    for (int i = 0; i < ffcell->num_weights; i++)
    {
        ffcell->weights[i / CHUNK_SIZE][i % CHUNK_SIZE] = frand() - 0.5;
    }
    ffcell->bias = frand() - 0.5;
}

// Returns random float in [0.0 - 1.0]
static float frand(void)
{
    return get_random() / (float)RAND_MAX;
}
