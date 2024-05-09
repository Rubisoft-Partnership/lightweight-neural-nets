#pragma once

#include <metrics.h>
#include <stdio.h>


/**
 * Logs the metrics to a CSV file with the following format:
 * epoch,accuracy,balanced_accuracy,average_precision,average_recall,average_f1_score, [confusion_matrix_items]
 *
 * @param file The pointer to the opened file to log the metrics.
 * @param metrics The metrics to be logged.
 * @param epoch The current epoch.
 */
void log_metrics(const FILE *file, Metrics metrics, const int epoch);



/**
 * Retrieves the metrics at a specific epoch from a CSV file.
 *
 * This function takes a file path and an epoch number as input and returns the metrics
 * recorded at that epoch from a CSV file. 
 * In case the file does not exist or the epoch is not found, the function returns a Metrics struct with all fields set to 0.
 *
 * @param file_path The path to the CSV file.
 * @param epoch The epoch number for which to retrieve the metrics.
 * @return The metrics recorded at the specified epoch.
 */
Metrics get_metrics_at_epoch(const char *file_path, const int epoch);

/**
 * Retrieves the last recorded metrics from a CSV file.
 *
 * This function reads a CSV file located at the specified file path and retrieves the last recorded metrics.
 * The metrics are stored in a structure of type Metrics, which contains various performance measures such as accuracy,
 * balanced accuracy, average precision, average recall, average F1 score, and a normalized confusion matrix.
 *
 * @param file_path The path to the CSV file.
 * @return The last recorded metrics from the CSV file. If the file does not exist or cannot be opened, an empty Metrics structure is returned.
 */
Metrics get_last_metrics(const char *file_path);

/**
 * Retrieves the current metrics from the specified file.
 *
 * @param file The file to read the metrics from.
 * @return The current metrics.
 */
Metrics get_current_metrics(FILE *file);

/**
 * Retrieves all metrics from a specified file.
 *
 * This function reads metrics from a CSV file located at the specified `file_path`.
 * It returns a pointer to a `Metrics` structure containing the retrieved metrics.
 * The number of metrics read is stored in the `num_metrics` parameter.
 *
 * @param file_path The path to the CSV file.
 * @param num_metrics A pointer to an integer that will store the number of metrics read.
 * @return A pointer to a `Metrics` structure containing the retrieved metrics.
 */


/**
 * Retrieves all metrics from a specified file.
 *
 * This function reads metrics from a CSV file located at the specified `file_path`.
 * It returns an array of `Metrics` structures containing the retrieved metrics.
 *
 * @param file_path The path to the CSV file.
 * @param num_metrics A pointer to an integer that will store the number of metrics read.
 * @return A dynamic array of `Metrics` structures containing the retrieved metrics.
 */
Metrics *get_all_metrics(const char *file_path, int *num_metrics);