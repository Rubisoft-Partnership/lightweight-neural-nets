/**
 * @file data.c
 * @brief This file contains the implementation of data-related functions.
 */

#include <stdlib.h>

#include <data/data.h>
#include <utils/utils.h>
#include <logging/logging.h>

/**
 * @brief Creates a new data object.
 *
 * @param feature_len The length of each input feature.
 * @param num_class The number of output classes.
 * @param rows The number of rows in the data object.
 * @return The newly created data object.
 */
Data new_data(const int feature_len, const int num_class, const int rows)
{
    const Data data = {
        new_matrix(rows, feature_len),
        new_matrix(rows, num_class),
        feature_len,
        num_class,
        rows};
    return data;
}

// TODO: adapt to contiguous memory allocation calling free once.
/**
 * @brief Frees a data object from the heap.
 *
 * @param data The data object to be freed.
 */
void free_data(const Data data)
{
    for (int row = 0; row < data.rows; row++)
    {
        free(data.input[row]);
        free(data.target[row]);
    }
    free(data.input);
    free(data.target);
}

/**
 * @brief Parses a string and extracts one row of inputs and outputs into the data object.
 *
 * @param data The data object to store the parsed values.
 * @param line The string containing the data values.
 * @param row The row index to store the values in.
 */
void parse_data(const Data data, char *line, const int row)
{
    const int cols = data.feature_len + data.num_class;
    for (int col = 0; col < cols; col++)
    {
        const double val = atof(strtok(col == 0 ? line : NULL, " "));
        if (col < data.feature_len)
            data.input[row][col] = val;
        else
            data.target[row][col - data.feature_len] = val;
    }
}

/**
 * @brief Randomly shuffles the rows of a data object.
 *
 * @param data The data object to be shuffled.
 */
void shuffle_data(const Data data)
{
    for (int a = 0; a < data.rows; a++)
    {
        const int b = get_random() % data.rows;
        double *ot = data.target[a];
        double *it = data.input[a];
        // Swap output.
        data.target[a] = data.target[b];
        data.target[b] = ot;
        // Swap input.
        data.input[a] = data.input[b];
        data.input[b] = it;
    }
}

/**
 * Creates a new batch of feedforward samples.
 *
 * @param size The size of the batch.
 * @return The newly created FFBatch.
 */
FFBatch new_ff_batch(const int batch_size, const int sample_size)
{
    log_debug("Creating batch object with size %d and sample size %d", batch_size, sample_size);
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
    log_debug("Freeing batch object");
    for (int i = 0; i < batch.size; i++)
    {
        free(batch.pos[i]);
        free(batch.neg[i]);
    }
    free(batch.pos);
    free(batch.neg);
}

/**
 * Generates positive and negative samples based on the given data and row index.
 * 
 * @param data The data structure containing input and target data.
 * @param row The index of the row to generate samples from.
 * @param pos Pointer to the array where the positive sample will be stored.
 * @param neg Pointer to the array where the negative sample will be stored.
 */
void generate_samples(const Data data, const int row, double *pos, double *neg)
{
    memcpy(pos, data.input[row], (data.feature_len - data.num_class) * sizeof(double));
    memcpy(neg, data.input[row], (data.feature_len - data.num_class) * sizeof(double));
    memcpy(&pos[data.feature_len - data.num_class], data.target[row], data.num_class * sizeof(double));
    memset(&neg[data.feature_len - data.num_class], 0, data.num_class * sizeof(double));
    // Set the positive sample's label to 0.0f
    int one_pos = -1;
    for (int i = data.feature_len - data.num_class; i < data.feature_len; i++)
        if (pos[i] == 1.0f)
            one_pos = i - (data.feature_len - data.num_class);
    // Generate a random label for the negative sample different from the positive sample's label
    int step = 1 + get_random() % (data.num_class - 1);
    int neg_label = (one_pos + step) % data.num_class;
    // Set the negative sample's label to 1.0f
    neg[(data.feature_len - data.num_class) + neg_label] = 1.0f;
}

void generate_batch(const Data data, const int batch_index, FFBatch batch)
{
    log_debug("Generating batch %d", batch_index);
    for (int i = 0; i < batch.size; i++)
    {
        const int index = (batch_index * batch.size + i) % data.rows;
        generate_samples(data, index, batch.pos[i], batch.neg[i]);
    }
}

/**
 * @brief Builds a data object by parsing a file and extracting the inputs and outputs for the neural network.
 *
 * @return The built data object.
 */
Data data_build(void)
{
    log_debug("Building data from %s", DATA_DATASET_PATH);
    FILE *file = fopen(DATA_DATASET_PATH, "r");
    if (file == NULL)
    {
        printf("Could not open %s\n", DATA_DATASET_PATH);
        exit(1);
    }
    const int rows = file_lines(file);
    Data data = new_data(DATA_FEATURES, DATA_CLASSES, rows);
    for (int row = 0; row < rows; row++)
    {
        char *line = read_line_from_file(file);
        parse_data(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}
