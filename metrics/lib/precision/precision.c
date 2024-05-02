/**
 * @file precision.c
 * @brief Implementation of the precision metric calculation.
 */

#include <precision/precision.h>

#include <stdio.h>

/**
 * Predictions struct that holds the true and predicted labels.
 * This struct is externed from predictions.c.
 */
extern Predictions predictions;

/**
 * @brief Calculates the average precision metric.
 *
 * This function calculates the precision for each class and returns the average precision.
 *
 * @return The average precision metric as a float value.
 */
float get_average_precision(void)
{
    // Initialize arrays to store true positives and false positives for each class
    int true_positives[NUM_CLASSES] = {0};  // Initialize all elements to 0
    int false_positives[NUM_CLASSES] = {0}; // Initialize all elements to 0

    // Count true positives and false positives for each class
    for (int i = 0; i < predictions.num_predictions; i++)
    {
        if (predictions.true_labels[i] == predictions.predicted_labels[i])
            true_positives[predictions.true_labels[i]]++;
        else
            false_positives[predictions.predicted_labels[i]]++;
    }

    // Calculate average precision
    float average_precision = 0.0;
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        if (true_positives[i] + false_positives[i] == 0)
            continue;
        average_precision += (float)true_positives[i] / (true_positives[i] + false_positives[i]);
    }
    return average_precision / NUM_CLASSES;
}

/**
 * @brief Calculates the precision metric for a specific class.
 *
 * This function calculates the precision for a specific class by counting the true positives and false positives.
 * `tp/(tp+fp)` where tp is the number of true positives and fp is the number of false positives.
 *
 * @param target_class The class for which to calculate the precision.
 *
 * @return The precision for the target class as a float value.
 */
float get_precision_for_class(Label target_class)
{
    int true_positives = 0;
    int false_positives = 0;
    for (int i = 0; i < predictions.num_predictions; i++)
    {
        if (predictions.predicted_labels[i] != target_class)
            continue;
        if (predictions.true_labels[i] == target_class)
            true_positives++;
        else
            false_positives++;
    }
    if (true_positives + false_positives == 0)
        return 0.0;
    return (float)true_positives / (true_positives + false_positives);
}
