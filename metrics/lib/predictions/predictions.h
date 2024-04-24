/**
 * @file predictions.h
 * @brief Header file containing the definition of the Predictions structure and related functions.
 */

#pragma once

#define NUM_CLASSES 10

#define MAX_PREDICTIONS 16384 /**< The maximum number of predictions that can be stored. */

/**
 * @brief Struct representing a set of predictions.
 */
typedef struct
{
    int true_labels[MAX_PREDICTIONS];      /**< The true labels for each prediction. */
    int predicted_labels[MAX_PREDICTIONS]; /**< The predicted labels for each prediction. */
    int num_predictions;                   /**< The number of predictions made. */
} Predictions;

/**
 * @brief Initializes the predictions structure by setting the number of predictions to 0.
 */
void init_predictions();

/**
 * @brief Adds a prediction to the predictions structure.
 *
 * @param true_label The true label of the prediction.
 * @param predicted_label The predicted label of the prediction.
 */
void add_prediction(const int true_label, const int predicted_label);