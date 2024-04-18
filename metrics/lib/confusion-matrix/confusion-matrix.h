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
void printConfusionMatrix();

/**
 * @brief Prints the normalized confusion matrix.
 *
 * This function prints the normalized confusion matrix based on the provided confusion matrix.
 * The normalized confusion matrix is obtained by dividing each cell value by the sum of the corresponding row,
 * which represents the total number of samples for that class.
 *
 * @param None
 */
void printNormalizedConfusionMatrix();