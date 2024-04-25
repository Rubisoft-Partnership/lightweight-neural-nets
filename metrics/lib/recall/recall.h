/**
 * @file recall.h
 * @brief This file contains the declarations of functions related to the recall metric.
 */

#pragma once

#include <predictions/predictions.h>

/**
 * @brief Calculates the average recall metric.
 *
 * This function calculates the recall for each class and returns the average recall.
 *
 * @return The average recall metric as a float value.
 */
float get_average_recall(void);

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
float get_recall_for_class(Label target_class);
