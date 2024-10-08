/**
 * @file confusion-matrix.c
 * @brief Implementation of functions for creating and printing confusion matrices.
 */

#include <confusion-matrix/confusion-matrix.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/**
 * @brief Creates a new confusion matrix based on the predictions->
 *
 * This function creates a new confusion matrix based on the true and predicted labels stored in the predictions struct.
 * The confusion matrix is a square matrix that represents the performance of a classification model.
 * Each cell in the matrix represents the number of samples that were predicted to belong to a certain class.
 *
 * @param None
 * @return The newly created confusion matrix.
 */
int **new_confusion_matrix(Predictions *predictions)
{
    int **confusionMatrix = (int **)calloc(NUM_CLASSES, sizeof(int *));
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        confusionMatrix[i] = (int *)calloc(NUM_CLASSES, sizeof(int));
    }
    for (int i = 0; i < predictions->num_predictions; i++)
    {
        Label true_label = predictions->true_labels[i];
        Label predicted_label = predictions->predicted_labels[i];
        confusionMatrix[true_label][predicted_label]++;
    }
    return confusionMatrix;
}

/**
 * Frees the memory allocated for a confusion matrix.
 *
 * @param confusionMatrix The confusion matrix to be freed.
 */
void free_confusion_matrix(int **confusionMatrix)
{
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        free(confusionMatrix[i]);
    }
    free(confusionMatrix);
}

/**
 * Frees the memory allocated for a normalized confusion matrix.
 *
 * This function takes a 2D array representing a normalized confusion matrix and frees the memory
 * allocated for it. The normalized confusion matrix is a square matrix where each element
 * represents the probability of predicting a certain class given the true class. The function
 * iterates over each row of the matrix and frees the memory allocated for it, and then frees the
 * memory allocated for the matrix itself.
 *
 * @param normalized_confusion_matrix The normalized confusion matrix to be freed.
 */
void free_normalized_confusion_matrix(float **normalized_confusion_matrix)
{
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        free(normalized_confusion_matrix[i]);
    }
    free(normalized_confusion_matrix);
}

/**
 * Calculates and returns the normalized confusion matrix.
 *
 * @return A 2D array of floats representing the normalized confusion matrix.
 */
float **get_normalized_confusion_matrix(Predictions *predictions)
{
    int **confusionMatrix = new_confusion_matrix(predictions);
    float **normalized_confusion_matrix = (float **)calloc(NUM_CLASSES, sizeof(float *));
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        normalized_confusion_matrix[i] = (float *)calloc(NUM_CLASSES, sizeof(float));
    }
    for (Label i = 0; i < NUM_CLASSES; i++)
    {
        int samples_per_class = 0;
        for (Label j = 0; j < NUM_CLASSES; j++)
        {
            samples_per_class += confusionMatrix[i][j];
        }
        for (Label j = 0; j < NUM_CLASSES; j++)
        {
            if (samples_per_class > 0)
            {
                normalized_confusion_matrix[i][j] = (float)confusionMatrix[i][j] / samples_per_class;
            }
            else
            {
                normalized_confusion_matrix[i][j] = 0.0;
            }
        }
    }
    return normalized_confusion_matrix;
}

/**
 * @brief Prints the confusion matrix.
 *
 * This function prints the confusion matrix based on the provided confusion matrix.
 * The confusion matrix is a square matrix that represents the performance of a classification model.
 * Each cell in the matrix represents the number of samples that were predicted to belong to a certain class.
 *
 * @param None
 */
void print_confusion_matrix(int **confusionMatrix)
{
    const int width = 7; // Adjust the cell width as needed

    printf("Confusion Matrix:\n");

    // Print top header
    printf("  *  ");
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        printf("|  %-*d", width - 3, i);
    }
    printf("|\n");

    // Print header separator
    printf("-----");
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        for (int j = 0; j < width; ++j)
            printf("-");
    }
    printf("|\n");

    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        // Print row label
        printf("  %d  | ", i);

        // Print row data
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            printf("%*d | ", width - 3, confusionMatrix[i][j]);
        }
        printf("\n");
    }

    // Print bottom separator
    printf("-----");
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        for (int j = 0; j < width; ++j)
            printf("-");
    }
    printf("|\n");
    free_confusion_matrix(confusionMatrix);
}

/**
 * @brief Prints the normalized confusion matrix.
 *
 * This function prints the normalized confusion matrix based on the provided confusion matrix.
 * The normalized confusion matrix is obtained by dividing each cell value by the sum of the corresponding row,
 * which represents the total number of samples for that class.
 *
 * @param None
 */
void print_normalized_confusion_matrix(float **confusionMatrix)
{
    const int width = 7; // Width of each cell in the table

    // Print top header
    printf("  *  ");
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        printf("|  %-*d", width - 3, i);
    }
    printf("|\n");

    // Print header separator
    printf("-----");
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        for (int j = 0; j < width; ++j)
            printf("-");
    }
    printf("|\n");

    for (int i = 0; i < NUM_CLASSES; ++i)
    {

        // Print row label
        printf("  %d  | ", i);

        // Print row data
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            printf("%-*.*f| ", width - 2, 2, (float)confusionMatrix[i][j]);
        }
        printf("\n");
    }

    // Print bottom separator
    printf("-----");
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        for (int j = 0; j < width; ++j)
            printf("-");
    }
    printf("|\n");
    free_normalized_confusion_matrix(confusionMatrix);
}
