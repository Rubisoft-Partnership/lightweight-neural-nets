#include <csv-logging/csv-logging.h>

#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 5000

/**
 * Logs the metrics to a CSV file with the following format:
 * epoch,accuracy,balanced_accuracy,average_precision,average_recall,average_f1_score, [confusion_matrix_items]
 *
 * @param file The pointer to the opened file to log the metrics.
 * @param metrics The metrics to be logged.
 * @param epoch The current epoch.
 */
void log_metrics(const FILE *file, Metrics metrics, const int epoch)
{
    // Print metrics
    fprintf(file, "%d,%f,%f,%f,%f,%f", epoch, metrics.accuracy, metrics.balanced_accuracy, metrics.average_precision, metrics.average_recall, metrics.average_f1_score);
    for (int i = 0; i < NUM_CLASSES; i++)
        for (int j = 0; j < NUM_CLASSES; j++)
            fprintf(file, "%f,", metrics.normalized_confusion_matrix[i][j]);
    fprintf(file, "\n");
}

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
Metrics get_metrics_at_epoch(const char *file_path, const int epoch)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        Metrics empty_metrics = {0};
        return empty_metrics;
    }

    Metrics metrics = {0};
    metrics.normalized_confusion_matrix = NULL;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file))
    {
        char *token = strtok(line, ",");
        if (atoi(token) == epoch)
        {
            metrics.accuracy = atof(strtok(NULL, ","));
            metrics.balanced_accuracy = atof(strtok(NULL, ","));
            metrics.average_precision = atof(strtok(NULL, ","));
            metrics.average_recall = atof(strtok(NULL, ","));
            metrics.average_f1_score = atof(strtok(NULL, ","));
            metrics.normalized_confusion_matrix = (float **)calloc(NUM_CLASSES, sizeof(float *));
            for (int i = 0; i < NUM_CLASSES; i++)
            {
                metrics.normalized_confusion_matrix[i] = (float *)calloc(NUM_CLASSES, sizeof(float));
                for (int j = 0; j < NUM_CLASSES; j++)
                    metrics.normalized_confusion_matrix[i][j] = atof(strtok(NULL, ","));
            }
            break;
        }
    }
    fclose(file);
    return metrics;
}

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
Metrics get_last_metrics(const char *file_path)
{
    Metrics metrics = {0};
    metrics.normalized_confusion_matrix = NULL;
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
        return metrics;

    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file))
    {
        strtok(line, ",");
        metrics.accuracy = atof(strtok(NULL, ","));
        metrics.balanced_accuracy = atof(strtok(NULL, ","));
        metrics.average_precision = atof(strtok(NULL, ","));
        metrics.average_recall = atof(strtok(NULL, ","));
        metrics.average_f1_score = atof(strtok(NULL, ","));
        metrics.normalized_confusion_matrix = (float **)calloc(NUM_CLASSES, sizeof(float *));
        for (int i = 0; i < NUM_CLASSES; i++)
        {
            metrics.normalized_confusion_matrix[i] = (float *)calloc(NUM_CLASSES, sizeof(float));
            for (int j = 0; j < NUM_CLASSES; j++)
                metrics.normalized_confusion_matrix[i][j] = atof(strtok(NULL, ","));
        }
    }
    fclose(file);
    return metrics;
}

/**
 * Retrieves the current metrics from the specified file.
 *
 * @param file The file to read the metrics from.
 * @return The current metrics.
 */
Metrics get_current_metrics(FILE *file)
{
    Metrics metrics = {0};
    metrics.normalized_confusion_matrix = NULL;
    char line[MAX_LINE_LENGTH];
    if (fgets(line, MAX_LINE_LENGTH, file))
    {
        strtok(line, ",");
        metrics.accuracy = atof(strtok(NULL, ","));
        metrics.balanced_accuracy = atof(strtok(NULL, ","));
        metrics.average_precision = atof(strtok(NULL, ","));
        metrics.average_recall = atof(strtok(NULL, ","));
        metrics.average_f1_score = atof(strtok(NULL, ","));
        metrics.normalized_confusion_matrix = (float **)calloc(NUM_CLASSES, sizeof(float *));
        for (int i = 0; i < NUM_CLASSES; i++)
        {
            metrics.normalized_confusion_matrix[i] = (float *)calloc(NUM_CLASSES, sizeof(float));
            for (int j = 0; j < NUM_CLASSES; j++)
                metrics.normalized_confusion_matrix[i][j] = atof(strtok(NULL, ","));
        }
    }
    return metrics;
}

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
Metrics *get_all_metrics(const char *file_path, int *num_metrics)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        return NULL;
    }

    // finds the number of lines in the file
    int lines = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, MAX_LINE_LENGTH, file))
    {
        lines++;
    }
    *num_metrics = lines;
    Metrics *metrics = (Metrics *)malloc(sizeof(Metrics) * (*num_metrics));

    // reads the file and stores the metrics in the metrics array
    fseek(file, 0, SEEK_SET);
    for (int i = 0; i < *num_metrics; i++)
        metrics[i] = get_current_metrics(file);
    

    fclose(file);

    return metrics;
}