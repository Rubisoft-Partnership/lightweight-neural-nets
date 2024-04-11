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
 * The FFNet struct contains an array of FFCell blocks, the number of cells, and the loss function suite.
 */
typedef struct
{
    FFCell layers[MAX_LAYERS_NUM];  ///< Array of FFCell blocks in the network.
    int num_cells;                  ///< Number of cells in the network.
    Loss loss_suite;                ///< Loss function suite for the network.
} FFNet;

/**
 * @brief Builds a new FFNet by creating multiple FFCell objects.
 *
 * This function creates a new FFNet by initializing the layers with the given layer sizes, activation function, and threshold.
 *
 * @param layer_sizes An array of integers representing the sizes of each layer in the network.
 * @param num_layers The number of layers in the network.
 * @param act The activation function for the cells in the network.
 * @param pdact The derivative of the activation function for the cells in the network.
 * @param threshold The threshold value for the cells in the network.
 * @param loss_suite The loss function suite for the network.
 * @return The newly created FFNet.
 */
FFNet new_ff_net(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double threshold, Loss loss_suite);

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
 * @param rate The learning rate for training.
 * @return The training loss.
 */
double train_ff_net(const FFNet ffnet, const double *const pos, const double *const neg, double rate);

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



