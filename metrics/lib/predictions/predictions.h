/**
 * @file predictions.h
 * @brief Header file containing the definition of the Predictions structure and related functions.
 */

#pragma once

#define NUM_CLASSES 10

#define MAX_PREDICTIONS 16384 /**< The maximum number of predictions that can be stored. */

typedef int Label;


/**
 * @brief Struct representing a set of predictions.
 */
typedef struct
{
    Label true_labels[MAX_PREDICTIONS];      /**< The true labels for each prediction. */
    Label predicted_labels[MAX_PREDICTIONS]; /**< The predicted labels for each prediction. */
    int num_predictions;                   /**< The number of predictions made. */
} Predictions;

/**
 * @brief Initializes the predictions structure by setting the number of predictions to 0.
 */
void init_predictions();

/**
 * @brief Resets the predictions.
 *
 * This function resets the predictions made by the neural network model.
 * After calling this function, the predictions will be cleared and ready
 * for new predictions.
 */
void reset_predictions();

/**
 * @brief Adds a prediction to the predictions structure.
 *
 * @param true_label The true label of the prediction.
 * @param predicted_label The predicted label of the prediction.
 */
void add_prediction(const Label true_label, const Label predicted_label);