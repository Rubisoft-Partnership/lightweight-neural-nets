/**
 * @file adam.h
 * @brief Header file for the Adam optimizer implementation.
 */

#pragma once

#include <string.h>
#include <stdlib.h>

/**
 * @struct Adam
 * @brief Struct representing the Adam optimizer.
 */
typedef struct
{
    double beta1; /**< Adam hyperparameter: exponential decay rate for the first moment estimate */
    double beta2; /**< Adam hyperparameter: exponential decay rate for the second moment estimate */
    double *m; /**< First moment estimate vector */
    double *v; /**< Second moment estimate vector */
    int t; /**< Time step */
} Adam;

/**
 * @brief Creates a new Adam optimizer instance.
 * @param beta1 The exponential decay rate for the first moment estimate.
 * @param beta2 The exponential decay rate for the second moment estimate.
 * @param size The size of the weight vector.
 * @return The created Adam optimizer instance.
 */
Adam adam_create(double beta1, double beta2, int size);

/**
 * @brief Frees the memory allocated for an Adam optimizer instance.
 * @param adam The Adam optimizer instance to be freed.
 */
void adam_free(Adam adam);

/**
 * @brief Updates the weight using the Adam optimizer.
 * @param adam The Adam optimizer instance.
 * @param gradient The gradient of the weight.
 * @param index The index of the weight.
 * @return The weight update value.
 */
double adam_weight_update(Adam adam, double gradient, int index);
