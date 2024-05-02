/**
 * @file precision.h
 * @brief This file contains the declarations of functions related to the precision metric.
 */

#pragma once

#include <predictions/predictions.h>

/**
 * @brief Calculates the average precision metric.
 *
 * This function calculates the precision for each class and returns the average precision.
 *
 * @return The average precision metric as a float value.
 */
float get_average_precision(void);

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
float get_precision_for_class(Label target_class);
