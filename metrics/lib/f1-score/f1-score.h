/**
 * @file f1-score.h
 * @brief This file contains the declarations of functions related to the f1-score metric.
 */

#pragma once

#include <predictions/predictions.h>

/**
 * @brief Calculates the average f1-score metric.
 *
 * This function calculates the f1-score for each class and returns the average f1-score.
 *
 * @return The average f1-score metric as a float value.
 */
float get_average_f1_score(void);

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
float get_f1_score_for_class(Label target_class);
