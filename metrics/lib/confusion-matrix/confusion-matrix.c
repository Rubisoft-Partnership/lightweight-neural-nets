#include "confusion-matrix.h"
#include <stdio.h>

#define NUM_CLASSES 10

// Confusion matrix storage
static int confusionMatrix[NUM_CLASSES][NUM_CLASSES];

// Initialize the confusion matrix
void initConfusionMatrix() {
    for (int i = 0; i < NUM_CLASSES; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            confusionMatrix[i][j] = 0;
        }
    }
}

// Add prediction data to the confusion matrix
void addPrediction(int true_label, int predicted_label) {
    if (true_label >= 0 && true_label < NUM_CLASSES && predicted_label >= 0 && predicted_label < NUM_CLASSES) {
        confusionMatrix[true_label][predicted_label]++;
    } else {
        printf("Error: Label indices out of bounds.\n");
    }
}

// Pretty print the normalized confusion matrix
void printNormalizedConfusionMatrix() {
    printf("Normalized Confusion Matrix:\n");
    printf("  ");
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("%5d", i);
    }
    printf("\n\n");
    
    for (int i = 0; i < NUM_CLASSES; ++i) {
        // Calculate the sum for normalization
        int sum = 0;
        for (int j = 0; j < NUM_CLASSES; ++j) {
            sum += confusionMatrix[i][j];
        }

        printf("%d   ", i);
        for (int j = 0; j < NUM_CLASSES; ++j) {
            if (sum > 0) {
                printf("%.2f ", (float)confusionMatrix[i][j] / sum);
            } else {
                printf("0.00 "); // Handle the case where sum is 0 to avoid division by zero
            }
        }
        printf("\n");
    }
}

// Pretty print the confusion matrix
void printConfusionMatrix() {
    printf("Confusion Matrix:\n");
    printf("  ");
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("%5d", i);
    }
    printf("\n\n");
    
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("%d   ", i);
        for (int j = 0; j < NUM_CLASSES; ++j) {
            printf("%5d", confusionMatrix[i][j]);
        }
        printf("\n");
    }
}