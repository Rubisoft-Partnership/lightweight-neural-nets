/**
 * @file accuracy.h
 * @brief This file contains the declarations of functions related to the accuracy metric.
 */

#pragma once

/**
 * @brief Initializes the accuracy metric.
 *
 * This function sets the number of correctly classified samples and the total number of samples to zero.
 */
void initAccuracy();

/**
 * @brief Adds a prediction to the accuracy metric.
 *
 * This function updates the number of correctly classified samples and the total number of samples based on the provided true label and predicted label.
 *
 * @param true_label The true label of the sample.
 * @param predicted_label The predicted label of the sample.
 */
void addPredictionAccuracy(int true_label, int predicted_label);

/**
 * @brief Calculates the accuracy metric.
 *
 * This function calculates the accuracy metric by dividing the number of correctly classified samples by the total number of samples.
 *
 * @return The accuracy metric as a double value.
 */
double getAccuracy();