/**
 * @file predictions.c
 * @brief Implementation of the Predictions module.
 *
 * The Predictions module provides functions for managing a collection of predictions.
 * It allows initializing the predictions structure, adding predictions to it, and retrieving information about the predictions.
 */
#include <predictions/predictions.h>

#include <stdio.h>

/**
 * @brief Declaration of the Predictions structure.
 *
 * The Predictions structure represents a collection of predictions.
 * It is used to store the predicted values generated by a neural network model.
 *
 * @see predictions.h
 */
Predictions predictions;

/**
 * @brief Initializes the predictions structure by setting the number of predictions to 0.
 */
void init_predictions()
{
    predictions.num_predictions = 0;
}

/**
 * @brief Adds a prediction to the predictions structure.
 *
 * This function adds a prediction to the predictions structure.
 * It takes the true label and the predicted label as parameters and stores them in the structure.
 * If the maximum number of predictions has been reached, it prints a message indicating that.
 *
 * @param true_label The true label of the prediction.
 * @param predicted_label The predicted label of the prediction.
 */
void add_prediction(const int true_label, const int predicted_label)
{
    if (predictions.num_predictions < MAX_PREDICTIONS)
    {
        predictions.true_labels[predictions.num_predictions] = true_label;
        predictions.predicted_labels[predictions.num_predictions] = predicted_label;
        predictions.num_predictions++;
    }
    else
    {
        printf("Maximum number of predictions reached.\n");
    }
}