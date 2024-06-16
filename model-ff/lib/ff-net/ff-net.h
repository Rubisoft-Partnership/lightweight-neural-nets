/**
 * @file ff-net.h
 * @brief Header file for the FFNet module.
 *
 * This file contains the definition of the FFNet struct and the function prototypes for building, training, and predicting with a forward forward neural network.
 */

#pragma once

#include <stdbool.h>

#include <ff-cell/ff-cell.h>
#include <losses/losses.h>

#define MAX_LAYERS_NUM 16

#define FFNET_CHECKPOINT_PATH PROJECT_BASEPATH "/checkpoints"

/**
 * @struct FFNet
 * @brief Struct that represents a forward forward neural network.
 *
 * The FFNet struct contains an array of FFCell blocks, the number of cells, the threshold value, and the loss function suite.
 */
typedef struct
{
    FFCell layers[MAX_LAYERS_NUM]; // Array of FFCell blocks in the network.
    int num_cells;                 // Number of cells in the network.
    double threshold;              // Threshold value for the cells in the network.
    LossType loss;                 // Loss function suite for the network.
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
FFNet *new_ff_net(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double threshold, const double beta1, const double beta2, LossType loss_suite);

/**
 * @brief Frees the memory allocated for a FFNet.
 *
 * This function frees the memory allocated for a FFNet, including the memory allocated for the FFCell objects.
 *
 * @param ffnet The FFNet to free.
 */
void free_ff_net(FFNet *ffnet);

/**
 * @brief Trains a FFNet by training each cell.
 *
 * This function trains a FFNet by training each cell in the network using the given positive and negative samples and learning rate.
 *
 * @param ffnet The FFNet to train.
 * @param batch Batch containing an array for positive samples and an array for negative samples.
 * @param learning_rate The learning rate for the training.
 * @return The training loss.
 */
double train_ff_net(FFNet *ffnet, const FFBatch batch, const double learning_rate);

/**
 * Calculates the loss on the given dataset and adds the predictions to the metrics.
 *
 * @param ffnet The FFNet model to test.
 * @param data The dataset to test the model on.
 * @return The average loss of the model on the dataset.
 */
double test_ff_net(FFNet *ffnet, Data *data, const int input_size);

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
int predict_ff_net(const FFNet *ffnet, const double *input, const int num_classes, const int input_size);

/**
 * @brief Saves a FFNet to a file.
 *
 * This function saves a FFNet to a file.
 *
 * @param ffnet The FFNet to save.
 * @param filename The name of the file to save the FFNet.
 */
void save_ff_net(const FFNet *ffnet, const char *filename, bool default_path);

/**
 * @brief Loads a FFNet from a file.
 *
 * This function loads a FFNet from a file.
 *
 * @param ffnet The FFNet to load.
 * @param filename The name of the file to load the FFNet.
 * @param act The activation function for the FFNet.
 * @param pdact The derivative of the activation function for the FFNet.
 * @param beta1 The beta1 value of the Adam optimizer.
 * @param beta2 The beta2 value of the Adam optimizer.
 */
void load_ff_net(FFNet *ffnet, const char *filename, double (*act)(double), double (*pdact)(double),
                 const double beta1, const double beta2);
