/**
 * @file recall.c
 * @brief Implementation of the recall metric calculation.
 */

#include <recall/recall.h>

#include <stdio.h>

/**
 * Predictions struct that holds the true and predicted labels.
 * This struct is externed from predictions.c.
 */
extern Predictions predictions;

/**
 * @brief Calculates the average recall metric.
 *
 * This function calculates the recall for each class and returns the average recall.
 *
 * @return The average recall metric as a float value.
 */
float get_average_recall(void)
{
    // Initialize arrays to store true positives and false negatives for each class
    int true_positives[NUM_CLASSES] = {0};  // Initialize all elements to 0
    int false_negatives[NUM_CLASSES] = {0}; // Initialize all elements to 0

    // Count true positives and false negatives for each class
    for (int i = 0; i < predictions.num_predictions; i++)
    {
        if (predictions.true_labels[i] == predictions.predicted_labels[i])
            true_positives[predictions.true_labels[i]]++;
        else
            false_negatives[predictions.true_labels[i]]++;
    }

    // Calculate average recall
    float average_recall = 0.0;
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        if (true_positives[i] + false_negatives[i] == 0)
            continue;
        average_recall += (float)true_positives[i] / (true_positives[i] + false_negatives[i]);
    }
    return average_recall / NUM_CLASSES;
}

/**
 * @brief Calculates the recall metric for a specific class.
 *
 * This function calculates the recall for a specific class by counting the true positives and false negatives.
 * `tp/(tp+fn)` where tp is the number of true positives and fn is the number of false negatives.
 *
 * @param target_class The class for which to calculate the recall.
 *
 * @return The recall for the target class as a float value.
 */
float get_recall_for_class(Label target_class)
{
    int true_positives = 0;
    int false_negatives = 0;
    for (int i = 0; i < predictions.num_predictions; i++)
    {
        if (predictions.predicted_labels[i] == target_class && predictions.true_labels[i] == target_class)
            true_positives++;
        else if (predictions.predicted_labels[i] != target_class && predictions.true_labels[i] == target_class)
            false_negatives++;
    }
    if (true_positives + false_negatives == 0)
        return 0.0;
    return (float)true_positives / (true_positives + false_negatives);
}
