/**
 * @file utils.c
 * @brief This file contains utility functions for the lightweight neural network model.
 *
 * This file includes the necessary header files and defines utility functions used in the model.
 * The functions in this file provide various utility operations such as file I/O, memory allocation, and random number generation.
 */

#include <stdlib.h>

#include <utils/utils.h>


/**
 * Finds the maximum integer value in an array.
 *
 * @param array The array of integers.
 * @param size The size of the array.
 * @return The maximum integer value in the array.
 */
int max_int(const int* array, const int size) {
    int max = array[0];
    for(int i = 1; i < size; i++) {
        if(array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

/**
 * @brief Return the number of lines in a file.
 *
 * This function counts the number of lines in a given file.
 *
 * @param file The file to count the lines from.
 * @return The number of lines in the file.
 */
int file_lines(FILE *const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while ((ch = getc(file)) != EOF)
    {
        if (ch == '\n')
            lines++;
        pc = ch;
    }
    if (pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

/**
 * @brief Read a line from a file.
 *
 * This function reads a line from a given file and returns it as a dynamically allocated string.
 *
 * @param file The file to read the line from.
 * @return The read line as a dynamically allocated string.
 */
char *read_line_from_file(FILE *const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char *line = (char *)malloc((size) * sizeof(char));
    while ((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if (reads + 1 == size)
            line = (char *)realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// TODO: enforce contiguous memory allocation.
/**
 * @brief Create a matrix of doubles.
 *
 * This function creates a matrix of doubles with the specified number of rows and columns.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return The created matrix.
 */
double **new_matrix(const int rows, const int cols)
{
    double **row = (double **)malloc((rows) * sizeof(double *));
    for (int r = 0; r < rows; r++)
        row[r] = (double *)malloc((cols) * sizeof(double));
    return row;
}

static int current_seed = 0;

/**
 * @brief Set the seed for random number generation.
 *
 * This function sets the seed for generating random numbers.
 *
 * @param seed The seed value to set.
 */
void set_seed(const int seed)
{
    current_seed = seed;
}

/**
 * @brief Generate a random number.
 *
 * This function generates a random number using the current seed value.
 *
 * @return The generated random number.
 */
int get_random(void)
{
    srand(current_seed);
    current_seed = rand();
    return rand();
}
