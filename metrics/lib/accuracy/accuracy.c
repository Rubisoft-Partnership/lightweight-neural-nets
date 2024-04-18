/**
 * @file accuracy.c
 * @brief Implementation of the accuracy metric calculation.
 */

#include <accuracy/accuracy.h>

#include <predictions/predictions.h>

#include <stdio.h>

/**
 * Predictions struct that holds the true and predicted labels.
 * This struct is externed from predictions.c.
 */
extern Predictions predictions;

/**
 * @brief Calculates the accuracy metric.
 *
 * This function calculates the accuracy metric by dividing the number of correctly classified samples by the total number of samples.
 *
 * @return The accuracy metric as a float value.
 */
float get_accuracy()
{
    int correct_predictions = 0;
    for (int i = 0; i < predictions.num_predictions; i++)
    {
        if (predictions.true_labels[i] == predictions.predicted_labels[i])
        {
            correct_predictions++;
        }
    }
    return (float)correct_predictions / predictions.num_predictions;
}
