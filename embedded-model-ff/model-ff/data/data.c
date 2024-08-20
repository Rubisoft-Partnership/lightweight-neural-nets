/**
 * @file data.c
 * @brief This file contains the implementation of data-related functions.
 */

#include <stdlib.h>

#include <data/data.h>
#include <utils/utils.h>


/**
 * Creates a new batch of feedforward samples.
 *
 * @param batch_size The size of the batch.
 * @param sample_size The size of each sample.
 * @return The newly created FFBatch.
 */
FFBatch new_ff_batch(const int batch_size, const int sample_size)
{
    const FFBatch batch = {
        new_matrix(batch_size, sample_size),
        new_matrix(batch_size, sample_size),
        batch_size};
    return batch;
}

/**
 * Frees the memory allocated for a batch of feedforward samples.
 *
 * @param batch The FFBatch to be freed.
 */
void free_ff_batch(FFBatch batch)
{
    for (int i = 0; i < batch.size; i++)
    {
        free(batch.pos[i]);
        free(batch.neg[i]);
    }
    free(batch.pos);
    free(batch.neg);
}
