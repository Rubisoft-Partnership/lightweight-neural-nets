/**
 * @file utils.h
 * @brief This file contains utility functions used in the project.
 */
#pragma once

#include <stdio.h>

/**
 * @brief Reads a line from a file.
 *
 * @param file The file to read from.
 * @return A dynamically allocated string containing the line read from the file.
 */
char *read_line_from_file(FILE *const file);

/**
 * @brief Creates a new matrix with the specified number of rows and columns.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return A dynamically allocated 2D array representing the matrix.
 */
double **new_matrix(const int rows, const int cols);

/**
 * @brief Sets the seed for the random number generator.
 *
 * @param seed The seed value to set.
 */
void set_seed(const int seed);

/**
 * @brief Generates a random integer.
 *
 * @return A random integer.
 */
int get_random(void);

/**
 * @brief Calculates the number of lines in a file.
 *
 * @param file The file to calculate the length of.
 * @return The number of lines in the file.
 */
int file_lines(FILE *const file);
