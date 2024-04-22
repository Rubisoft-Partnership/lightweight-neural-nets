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

// Default project basepath is current directory.
#ifndef PROJECT_BASEPATH
#define PROJECT_BASEPATH ""
#endif
#define DATA_DATASET_BASEPATH PROJECT_BASEPATH "/../dataset/"

// Each dataset must have a folder with the dataset name containing the following files:
// - train.txt: the training data split.
// - test.txt: the testing data split.
#define DATA_TRAIN_SPLIT "train.txt"
#define DATA_TEST_SPLIT "test.txt"
#define DATA_VALIDATION_SPLIT "validation.txt"

#define DATA_MNIST_CLASSES 10
#define DATA_MNIST_FEATURES 784
#define DATA_MNIST_PATH "mnist/"

#define DATA_DIGITS_CLASSES 10
#define DATA_DIGITS_FEATURES 74
#define DATA_DIGITS_PATH "digits/"

#ifdef DATA_MNIST
#define DATA_CLASSES DATA_MNIST_CLASSES
#define DATA_FEATURES DATA_MNIST_FEATURES
#define DATA_TRAIN_PATH DATA_DATASET_BASEPATH DATA_MNIST_PATH DATA_TRAIN_SPLIT
#define DATA_TEST_PATH DATA_DATASET_BASEPATH DATA_MNIST_PATH DATA_TEST_SPLIT
#define DATA_VALIDATION_PATH DATA_DATASET_BASEPATH DATA_MNIST_PATH DATA_VALIDATION_SPLIT
#endif

#ifdef DATA_DIGITS
#define DATA_CLASSES DATA_DIGITS_CLASSES
#define DATA_FEATURES DATA_DIGITS_FEATURES
#define DATA_DATASET_PATH DATA_DATASET_BASEPATH DATA_DIGITS_PATH
#define DATA_TRAIN_PATH DATA_DATASET_BASEPATH DATA_DIGITS_PATH DATA_TRAIN_SPLIT
#define DATA_TEST_PATH DATA_DATASET_BASEPATH DATA_DIGITS_PATH DATA_TEST_SPLIT
#define DATA_VALIDATION_PATH DATA_DATASET_BASEPATH DATA_DIGITS_PATH DATA_VALIDATION_SPLIT
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

/**
 * @brief Creates a new data object.
 * 
 * @param feature_len Number of inputs to the neural network.
 * @param num_class Number of outputs to the neural network.
 * @param rows Number of rows in the file (number of sets for neural network).
 * @return Data The newly created data object.
 */
Data *new_data(const int feature_len, const int num_class, const int rows);

/**
 * @brief Frees the memory of a data object.
 * 
 * @param data The data object to free.
 */
void free_data(Data *data);

/**
 * @brief Builds a data object from a file.
 * 
 * @return Data The built data object.
 */
Data *data_build(const char *file_path);

/**
 * @brief Parses a line from a file into a data object.
 * 
 * @param data The data object to parse into.
 * @param line The line to parse.
 * @param row The row index of the data object.
 */
void parse_data(Data *data, char *line, const int row);

/**
 * @brief Shuffles the data object.
 * 
 * @param data The data object to shuffle.
 */
void shuffle_data(Data *data);

/**
 * @brief Creates a new FFsamples object.
 * 
 * @param input_size The size of the input.
 * @return FFsamples The newly created FFsamples object.
 */
FFsamples new_ff_samples(const int input_size);

/**
 * @brief Frees the memory of a FFsamples object.
 * 
 * @param samples The FFsamples object to free.
 */
void free_ff_samples(FFsamples samples);

/**
 * @brief Generates a positive and a negative sample for the FF algorithm by embedding the one-hot encoded target in the input.
 * 
 * @param data The data object.
 * @param row The row index of the data object.
 * @param samples The FFsamples object to store the generated samples.
 */
void generate_samples(const Data *data, const int row, FFsamples samples);

/**
 * @brief Structure representing a dataset.
 *
 * This structure holds pointers to the training, testing, and optional validation data.
 */
typedef struct
{
    Data *train;      /**< Pointer to the training data */
    Data *test;       /**< Pointer to the testing data */
    Data *validation; /**< Pointer to the validation data (optional) */
} Dataset;

/**
 * @brief Splits the dataset into training, testing, and optional validation data.
 *
 * @return The dataset structure containing the split data.
 */
Dataset dataset_split(void);

/**
 * @brief Frees the memory allocated for the dataset.
 *
 * @param dataset The dataset to be freed.
 */
void free_dataset(Dataset dataset);
