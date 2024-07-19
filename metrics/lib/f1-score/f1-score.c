/**
 * @file f1-score.c
 * @brief Implementation of the f1-score metric calculation.
 */

#include <f1-score/f1-score.h>

#include <stdio.h>


/**
 * @brief Calculates the average f1-score metric.
 *
 * This function calculates the f1-score for each class and returns the average f1-score.
 *
 * @return The average f1-score metric as a float value.
 */
float get_average_f1_score(Predictions *predictions)
{
    // Initialize arrays to store true positives and false negatives for each class
    int true_positives[NUM_CLASSES] = {0};  // Initialize all elements to 0
    int false_positives[NUM_CLASSES] = {0}; // Initialize all elements to 0
    int false_negatives[NUM_CLASSES] = {0}; // Initialize all elements to 0

    // Count true positives and false negatives for each class
    for (int i = 0; i < predictions->num_predictions; i++)
    {
        if (predictions->true_labels[i] == predictions->predicted_labels[i])
            true_positives[predictions->true_labels[i]]++;
        else
        {
            false_negatives[predictions->true_labels[i]]++;
            false_positives[predictions->predicted_labels[i]]++;
        }
    }

    // Calculate average f1-score
    float average_f1_score = 0.0;
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        if (true_positives[i] + false_negatives[i] + false_positives[i] == 0)
            continue;
        average_f1_score += (float)2.0 * true_positives[i] / (2.0 * true_positives[i] + false_positives[i] + false_negatives[i]);
    }
    return average_f1_score / NUM_CLASSES;
}

/**
 * @brief Calculates the f1-score metric for a specific class.
 *
 * This function calculates the f1-score for a specific class by counting the true positives and false negatives.
 * `2*tp/(2*tp+fp+fn)` where tp is the number of true positives, fn is the number of false negatives and fp is the number of false positives.
 *
 * @param target_class The class for which to calculate the f1-score.
 *
 * @return The f1-score for the target class as a float value.
 */
float get_f1_score_for_class(Predictions *predictions, Label target_class)
{
    int true_positives = 0;
    int false_negatives = 0;
    int false_positives = 0;
    for (int i = 0; i < predictions->num_predictions; i++)
    {
        if (predictions->true_labels[i] != target_class && predictions->predicted_labels[i] != target_class)
            continue;
        if (predictions->true_labels[i] == predictions->predicted_labels[i])
            true_positives++;
        else if (predictions->true_labels[i] == target_class)
            false_negatives++;
        else
            false_positives++;
    }
    if (true_positives + false_negatives + false_positives == 0)
        return 0.0;

    return (float)2.0 * true_positives / (2.0 * true_positives + false_positives + false_negatives);
}
