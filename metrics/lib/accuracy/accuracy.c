/**
 * @file accuracy.c
 * @brief Implementation of the accuracy metric calculation.
 */

#include "accuracy.h"

#include <stdio.h>

int correctly_classified = 0; /** Number of correctly classified samples. */
int total_samples = 0;        /** Total number of samples. */

/**
 * @brief Initializes the accuracy metric.
 *
 * This function sets the number of correctly classified samples and the total number of samples to zero.
 */
void initAccuracy()
{
    correctly_classified = 0;
    total_samples = 0;
}

/**
 * @brief Adds a prediction to the accuracy metric.
 *
 * This function updates the number of correctly classified samples and the total number of samples based on the provided true label and predicted label.
 *
 * @param true_label The true label of the sample.
 * @param predicted_label The predicted label of the sample.
 */
void addPredictionAccuracy(int true_label, int predicted_label)
{
    if (true_label == predicted_label)
        correctly_classified++;
    total_samples++;
}

/**
 * @brief Calculates the accuracy metric.
 *
 * This function calculates the accuracy metric by dividing the number of correctly classified samples by the total number of samples.
 *
 * @return The accuracy metric as a double value.
 */
double getAccuracy()
{
    return (double)correctly_classified / total_samples;
}
