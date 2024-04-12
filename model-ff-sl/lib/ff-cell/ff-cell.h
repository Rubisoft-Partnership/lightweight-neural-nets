/**
 * @file ff-cell.h
 * @brief Header file for the FFCell struct and related functions.
 */

#pragma once

#include <stdio.h>
#include <adam/adam.h>
#include <losses/losses.h>

/**
 * @def H_BUFFER_SIZE
 * @brief Size of buffer to store output activations.
 */
#define H_BUFFER_SIZE 1024

/**
 * @def MAX_CLASSES
 * @brief Maximum number of classes.
 */
#define MAX_CLASSES 16

/**
 * @struct FFCell
 * @brief FFCell struct that contains the weights, bias, and output layer.
 */
typedef struct
{
    double *weights;    /**< All the weights. */
    double bias;        /**< Biases. */
    double *output;     /**< Output layer. */
    int num_weights;             /**< Number of weights. */
    int input_size;     /**< Number of inputs. */
    int output_size;    /**< Number of outputs. */
    double (*act)(const double);     /**< Activation function. */
    double (*pdact)(const double);   /**< Derivative of activation function. */
    Adam adam;          /**< Adam optimizer. */
} FFCell;


/**
 * @brief Generates a new FFCell.
 * @param input_size The number of inputs.
 * @param output_size The number of outputs.
 * @param act The activation function.
 * @param pdact The derivative of the activation function.
 * @param threshold The hyperparameter for the FF algorithm.
 * @return The newly generated FFCell.
 */
FFCell new_ff_cell(const int input_size, const int output_size, double (*act)(double), double (*pdact)(double));

/**
 * @brief Frees the memory of a FFCell.
 * @param ffcell The FFCell to be freed.
 */
void free_ff_cell(FFCell ffcell);

/**
 * @brief Trains a FFCell by performing forward and backward pass with a given loss function.
 * @param ffcell The FFCell to be trained.
 * @param pos The positive samples.
 * @param neg The negative samples.
 * @param rate The learning rate.
 * @param loss_suite The loss function suite.
 * @return The loss value after training.
 */
double train_ff_cell(const FFCell ffcell, const double *const pos, const double *const neg, double rate, const double threshold, const Loss loss_suite);

/**
 * @brief Performs the forward pass for a FFCell.
 * @param ffcell The FFCell.
 * @param in The input values.
 */
void fprop_ff_cell(const FFCell ffcell, const double *const in);

/**
 * @brief Activation function: Rectified Linear Unit (ReLU).
 * @param a The input value.
 * @return The output value after applying the ReLU activation function.
 */
double relu(const double a);

/**
 * @brief Derivative of the ReLU activation function.
 * @param a The input value.
 * @return The derivative of the ReLU activation function at the given input value.
 */
double pdrelu(const double a);