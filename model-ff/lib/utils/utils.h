/**
 * @file utils.h
 * @brief This file contains utility functions used in the project.
 */
#pragma once

#include <stdio.h>

/**
 * @brief The width of the progress bar.
 */
#define PROGRESS_BAR_WIDTH 50

/**
 * Finds the maximum integer value in an array.
 *
 * @param array The array of integers.
 * @param size The size of the array.
 * @return The maximum integer value in the array.
 */
int max_int(const int* array, const int size);

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

/**
 * @brief Initializes the progress bar.
 */
void init_progress_bar(void);

/**
 * @brief Updates the progress bar with the current batch index and size.
 *
 * @param batch_index The current batch index.
 * @param batch_size The total number of batches.
 */
void update_progress_bar(const int batch_index, const int batch_size);

/**
 * @brief Finishes the progress bar.
 */
void finish_progress_bar(void);

/**
 * @brief Prints the elapsed time in a human-readable format.
 *
 * @param seconds_elapsed The number of seconds elapsed.
 */
void print_elapsed_time(const int seconds_elapsed);
