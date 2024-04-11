#pragma once

#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Set digits as default dataset.
#if !defined(DATA_MNIST) && !defined(DATA_DIGITS)
    #define DATA_DIGITS
#endif

#define DATA_DATASET_BASEPATH PROJECT_BASEPATH "/../dataset/"

#define DATA_MNIST_CLASSES 10
#define DATA_MNIST_FEATURES 784
#define DATA_MNIST_PATH "mnist/mnist_train.txt"

#define DATA_DIGITS_CLASSES 10
#define DATA_DIGITS_FEATURES 74
#define DATA_DIGITS_PATH "digits/digits.txt"

#ifdef DATA_MNIST
#define DATA_CLASSES DATA_MNIST_CLASSES
#define DATA_FEATURES DATA_MNIST_FEATURES
#define DATA_DATASET_PATH DATA_DATASET_BASEPATH DATA_MNIST_PATH
#endif

#ifdef DATA_DIGITS
#define DATA_CLASSES DATA_DIGITS_CLASSES
#define DATA_FEATURES DATA_DIGITS_FEATURES
#define DATA_DATASET_PATH DATA_DATASET_BASEPATH DATA_DIGITS_PATH
#endif

// Data object.
typedef struct
{
    // 2D floating point array of input.
    double **input;
    // 2D floating point array of target.
    double **target;
    // Number of inputs to neural network.
    int feature_len;
    // Number of outputs to neural network.
    int num_class;
    // Number of rows in file (number of sets for neural network).
    int rows;
} Data;

// FFsamples object.
typedef struct
{
    // floating point array of FF positive sample <input, correct_label>
    double *pos;
    // floating point array of FF negative sample <input, incorrect_label>
    double *neg;
} FFsamples;

// Creates a new data object.
Data new_data(const int feature_len, const int num_class, const int rows);
// Frees the memory of a data object.
void free_data(const Data d);
// Builds a data object from a file.
Data data_build(void);
// Parses a line from a file into a data object.
void parse_data(const Data data, char *line, const int row);
// Shuffles the data object.
void shuffle_data(const Data d);
// Creates a new FFsamples object.
FFsamples new_ff_samples(const int nips);
// Frees the memory of a FFsamples object.
void free_ff_samples(FFsamples s);
// Generates a positive and a negative sample for the FF algorithm by embedding the one-hot encoded target in the input.
void generate_samples(const Data d, const int row, FFsamples s);
