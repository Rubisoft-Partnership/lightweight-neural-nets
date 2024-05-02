/**
 * @file confusion-matrix.h
 * @brief Header file containing the declaration of functions for printing the confusion matrix.
 */
#pragma once

/**
 * @brief Prints the confusion matrix.
 *
 * This function prints the confusion matrix based on the provided confusion matrix.
 * The confusion matrix is a square matrix that represents the performance of a classification model.
 * Each cell in the matrix represents the number of samples that were predicted to belong to a certain class.
 *
 * @param None
 */
void print_confusion_matrix();

/**
 * @brief Prints the normalized confusion matrix.
 *
 * This function prints the normalized confusion matrix based on the provided confusion matrix.
 * The normalized confusion matrix is obtained by dividing each cell value by the sum of the corresponding row,
 * which represents the total number of samples for that class.
 *
 * @param None
 */
void print_normalized_confusion_matrix();

/**
 * Calculates and returns the normalized confusion matrix.
 * 
 * @return A 2D array of floats representing the normalized confusion matrix.
 */
float** get_normalized_confusion_matrix(void);


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
void free_normalized_confusion_matrix(float **normalized_confusion_matrix);
