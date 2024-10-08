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


// Each dataset must have a folder with the dataset name containing the following files:
// - train.txt: the training data split.
// - test.txt: the testing data split.
// - validation.txt: the validation data split (optional).
#define DATA_TRAIN_SPLIT "train.txt"
#define DATA_TEST_SPLIT "test.txt"
#define DATA_VALIDATION_SPLIT "validation.txt"

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
Data *new_data(const int feature_len, const int num_class, const int rows);

/**
 * @brief Frees the memory of a data object.
 *
 * @param data The data object to free.
 */
void free_data(Data *data);


/**
 * @brief Creates a new data object from a file.
 *
 * @param file_path The path to the file containing the data.
 * @param num_features The number of features in the dataset.
 * @param num_classes The number of classes in the dataset.
 *
 * @return The data object created from the file.
 */
Data *data_build(const char *file_path, const int num_features, const int num_classes);

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
 * @brief Creates a new batch of feedforward samples.
 *
 * @param batch_size The size of the batch.
 * @param sample_size The size of each sample.
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
void generate_samples(const Data *data, const int row, double *pos, double *neg);

/**
 * @brief Generates a batch of feedforward samples.
 *
 * @param data The data object.
 * @param row The row index of the data object.
 * @param batch The FFBatch object to store the generated samples.
 */
void generate_batch(const Data *data, const int row, FFBatch batch);

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
 * @param dataset_basepath The base path of the dataset, containing the training, testing, and optional validation data.
 * @param num_classes The number of classes in the dataset.
 *
 * @return The dataset structure containing the split data.
 */
Dataset dataset_split(const char *dataset_basepath, const int num_classes);

/**
 * @brief Frees the memory allocated for the dataset.
 *
 * @param dataset The dataset to be freed.
 */
void free_dataset(Dataset dataset);
