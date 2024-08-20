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

