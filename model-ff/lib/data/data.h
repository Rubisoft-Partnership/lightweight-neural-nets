#pragma once

#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Set digits as default dataset.
#if !defined(DATA_MNIST) && !defined(DATA_DIGITS)
    #define DATA_DIGITS
#endif

#define DATA_DATASET_BASEPATH "../../dataset/"

#define DATA_MNIST_CLASSES 10
#define DATA_MNIST_FEATURES 784
#define DATA_MNIST_PATH "mnist/mnist_train.txt"

#define DATA_DIGITS_CLASSES 10
#define DATA_DIGITS_FEATURES 64
#define DATA_DIGITS_PATH "digits/digits.txt"

#ifdef DATA_MNIST
#define DATA_CLASSES DATA_MNIST_CLASSES
#define DATA_FEATURES DATA_MNIST_FEATURES
#define DATA_DATASET_PATH (DATA_DATASET_BASEPATH DATA_MNIST_PATH)
#endif

#ifdef DATA_DIGITS
#define DATA_CLASSES DATA_DIGITS_CLASSES
#define DATA_FEATURES DATA_DIGITS_FEATURES
#define DATA_DATASET_PATH (DATA_DATASET_BASEPATH DATA_DIGITS_PATH)
#endif

// Data object.
typedef struct
{
    // 2D floating point array of input.
    double **in;
    // 2D floating point array of target.
    double **tg;
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

Data ndata(const int feature_len, const int num_class, const int rows);
void parse(const Data data, char *line, const int row);
void dfree(const Data d);
void shuffle(const Data d);
FFsamples new_samples(const int nips);
void generate_samples(const Data d, const int row, FFsamples s);
Data build(void);
