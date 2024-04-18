/**
 * @file ff-net.h
 * @brief Header file for the FFNet module.
 *
 * This file contains the definition of the FFNet struct and the function prototypes for building, training, and predicting with a forward forward neural network.
 */

#pragma once

#include <ff-cell/ff-cell.h>
#include <losses/losses.h>

#define MAX_LAYERS_NUM 64

/**
 * @struct FFNet
 * @brief Struct that represents a forward forward neural network.
 *
 * The FFNet struct contains an array of FFCell blocks, the number of cells, the threshold value, and the loss function suite.
 */
typedef struct
{
    FFCell layers[MAX_LAYERS_NUM]; ///< Array of FFCell blocks in the network.
    int num_cells;                 ///< Number of cells in the network.
    double threshold;              ///< Threshold value for the cells in the network.
    Loss loss_suite;               ///< Loss function suite for the network.
} FFNet;

/**
 * @brief Builds a FFNet by creating multiple FFCell objects.
 *
 * This function constructs a feedforward neural network (FFNet) by creating multiple FFCell objects.
 * The FFNet is built based on the provided layer sizes, activation function, derivative of the activation function,
 * threshold value, beta1, beta2, and loss function suite.
 *
 * @param layer_sizes The array of layer sizes, including the number of input and output units.
 * @param num_layers The number of layers in the FFNet.
 * @param act The activation function for the FFNet.
 * @param pdact The derivative of the activation function for the FFNet.
 * @param treshold The threshold value for the FFNet.
 * @param beta1 The beta1 value of the Adam optimizer\.
 * @param beta2 The beta2 value of the Adam optimizer\.
 * @param loss_suite The loss function suite for the FFNet.
 * @return FFNet The constructed FFNet.
 */
FFNet new_ff_net(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double threshold, double beta1, double beta2, Loss loss_suite);

/**
 * @brief Frees the memory allocated for a FFNet.
 *
 * This function frees the memory allocated for a FFNet, including the memory allocated for the FFCell objects.
 *
 * @param ffnet The FFNet to free.
 */
void free_ff_net(FFNet ffnet);

/**
 * @brief Trains a FFNet by training each cell.
 *
 * This function trains a FFNet by training each cell in the network using the given positive and negative samples and learning rate.
 *
 * @param ffnet The FFNet to train.
 * @param pos Array of the positive sample.
 * @param neg Array of the negative sample.
 * @param learning_rate The learning rate for the training.
 * @return The training loss.
 */
double train_ff_net(const FFNet ffnet, const double *const pos, const double *const neg, const double learning_rate);

/**
 * @brief Performs inference with a FFNet.
 *
 * This function performs inference with a FFNet by predicting the class label for the given input.
 *
 * @param ffnet The FFNet to use for inference.
 * @param input The input data.
 * @param num_classes The number of classes.
 * @param input_size The size of the input data.
 * @return The predicted class label.
 */
int predict_ff_net(const FFNet ffnet, const double *input, const int num_classes, const int input_size);
