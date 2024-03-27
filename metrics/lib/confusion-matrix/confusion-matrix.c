#include "confusion-matrix.h"
#include <stdio.h>

#define NUM_CLASSES 10

// Confusion matrix storage
static int confusionMatrix[NUM_CLASSES][NUM_CLASSES];

// Initialize the confusion matrix
void initConfusionMatrix()
{
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            confusionMatrix[i][j] = 0;
        }
    }
}

// Add prediction data to the confusion matrix
void addPrediction(int true_label, int predicted_label)
{
    if (true_label >= 0 && true_label < NUM_CLASSES && predicted_label >= 0 && predicted_label < NUM_CLASSES)
    {
        confusionMatrix[true_label][predicted_label]++;
    }
    else
    {
        printf("Error: Label indices out of bounds.\n");
    }
}

// Pretty print the normalized confusion matrix
void printNormalizedConfusionMatrix()
{
    int width = 7; // Width of each cell in the table
    int sample_num = 0;
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        for (int j = 0; j < NUM_CLASSES; j++)
        {
            sample_num += confusionMatrix[i][j];
        }
    }
    printf("Normalized confusion Matrix with %d samples:\n", sample_num);

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
        // Calculate sum for normalization
        int sum = 0;
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            sum += confusionMatrix[i][j];
        }

        // Print row label
        printf("  %d  | ", i);

        // Print row data
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            if (sum > 0)
            {
                printf("%-*.*f| ", width - 2, 2, (float)confusionMatrix[i][j] / sum);
            }
            else
            {
                printf("%-*s| ", width - 2, "0.00");
            }
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
}

void printConfusionMatrix()
{
    int width = 7; // Adjust the cell width as needed

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
}