/**
 * @file data.h
 * @brief Header file for data handling in the lightweight neural network model.
 *
 * This file contains the declarations for data handling functions and structures used in the model.
 * It provides functions for loading, preprocessing, and manipulating data for training and testing the neural network.
 */

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

// FFBatch object.
typedef struct
{
    // 2D floating point array of FF positive sample <input, correct_label>
    double **pos;
    // 2D floating point array of FF negative sample <input, incorrect_label>
    double **neg;
    // Number of samples in the batch.
    int size;
} FFBatch;

/**
 * @brief Creates a new data object.
 *
 * @param feature_len Number of inputs to the neural network.
 * @param num_class Number of outputs to the neural network.
 * @param rows Number of rows in the file (number of sets for neural network).
 * @return Data The newly created data object.
 */
Data new_data(const int feature_len, const int num_class, const int rows);

/**
 * @brief Frees the memory of a data object.
 *
 * @param data The data object to free.
 */
void free_data(const Data data);

/**
 * @brief Builds a data object from a file.
 *
 * @return Data The built data object.
 */
Data data_build(void);

/**
 * @brief Parses a line from a file into a data object.
 *
 * @param data The data object to parse into.
 * @param line The line to parse.
 * @param row The row index of the data object.
 */
void parse_data(const Data data, char *line, const int row);

/**
 * @brief Shuffles the data object.
 *
 * @param data The data object to shuffle.
 */
void shuffle_data(const Data data);

/**
 * @brief Creates a new batch of feedforward samples.
 *
 * @param size The size of the batch.
 * @return FFBatch The newly created FFBatch object.
 */
FFBatch new_ff_batch(const int batch_size, const int sample_size);

/**
 * @brief Frees the memory allocated for a batch of feedforward samples.
 *
 * @param batch The FFBatch object to free.
 */
void free_ff_batch(const FFBatch batch);

/**
 * @brief Generates a positive and a negative sample for the FF algorithm.
 *
 * @param data The data object.
 * @param row The row index of the data object.
 * @param pos The positive sample.
 * @param neg The negative sample.
 */
void generate_samples(const Data data, const int row, double *pos, double *neg);

/**
 * @brief Generates a positive and a negative sample for the FF algorithm by embedding the one-hot encoded target in the input.
 *
 * @param data The data object.
 * @param row The row index of the data object.
 * @param batch The FFBatch object to store the generated samples.
 */
void generate_batch(const Data data, const int row, FFBatch batch);
