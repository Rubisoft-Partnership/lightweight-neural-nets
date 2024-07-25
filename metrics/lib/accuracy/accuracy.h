/**
 * @file accuracy.h
 * @brief This file contains the declarations of functions related to the accuracy metric.
 */

#pragma once

#include <predictions/predictions.h>

/**
 * @brief Calculates the accuracy metric.
 *
 * This function calculates the accuracy metric by dividing the number of correctly classified samples by the total number of samples.
 *
 * @return The accuracy metric as a double value.
 */
float get_accuracy(Predictions *predictions);

/**
 * Calculates the balanced accuracy of a set of predictions.
 * 
 * The balanced accuracy is a metric that measures the performance of a binary classifier.
 * It takes into account both the sensitivity (true positive rate) and specificity (true negative rate)
 * of the classifier, and returns their average.
 * 
 * @return The balanced accuracy value.
 */
float get_balanced_accuracy(Predictions *predictions);
