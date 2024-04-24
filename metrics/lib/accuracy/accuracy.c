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
float get_accuracy(void)
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

/**
 * Calculates the balanced accuracy of a set of predictions.
 *
 * The balanced accuracy is a metric that measures the performance of a binary classifier.
 * It takes into account both the sensitivity (true positive rate) and specificity (true negative rate)
 * of the classifier, and returns their average.
 *
 * @return The balanced accuracy value.
 */
float get_balanced_accuracy(void)
{
    int correct_predictions[NUM_CLASSES];
    // Initialize correct predictions to the number of predictions for each class
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        correct_predictions[i] = predictions.num_predictions;
    }

    for (int i = 0; i < predictions.num_predictions; i++)
    {
        // If the prediction is incorrect, decrement the correct predictions for both the true and predicted labels
        if (predictions.true_labels[i] != predictions.predicted_labels[i])
        {
            correct_predictions[predictions.true_labels[i]]--;
            correct_predictions[predictions.predicted_labels[i]]--;
        }
    }
    int sum = 0;
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        sum += correct_predictions[i];
    }
    return (float)sum / (NUM_CLASSES * predictions.num_predictions);
}
