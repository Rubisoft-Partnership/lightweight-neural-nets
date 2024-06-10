/**
 * @file data.c
 * @brief This file contains the implementation of data-related functions.
 */

#include <stdlib.h>

#include <data/data.h>
#include <utils/utils.h>
#include <logging/logging.h>

/**
 * @brief The feature length of the data.
 *
 * @param file_path The path to the file containing the data.
 * @param num_classes The number of classes in the dataset.
 * 
 * @return The length of the features in the data.
 */
int get_feature_len(const char *file_path, const int num_classes)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        log_error("Could not open %s for feature len calculation", file_path);
        exit(EXIT_FAILURE);
    }
    char c = fgetc(file);
    int line_len = 1;
    while (c != '\n')
    {
        if (c == ' ')

            line_len++;
        c = fgetc(file);
    }
    return line_len - num_classes;
}

/**
 * @brief Splits the dataset into training, testing, and optional validation data.
 *
 * @param dataset_basepath The base path of the dataset, containing the training, testing, and optional validation data.
 * @param num_classes The number of classes in the dataset.
 *
 * @return The dataset structure containing the split data.
 */
Dataset dataset_split(const char *dataset_basepath, const int num_classes)
{
    // Buffer to store the full path of the dataset files.
    char full_path[256];
    Dataset dataset;
    sprintf(full_path, "%s/%s", dataset_basepath, DATA_TEST_SPLIT);
    const int feature_size = get_feature_len(full_path, num_classes);
    dataset.test = data_build(full_path, feature_size, num_classes);
    sprintf(full_path, "%s/%s", dataset_basepath, DATA_TRAIN_SPLIT);
    dataset.train = data_build(full_path, feature_size, num_classes);
    sprintf(full_path, "%s/%s", dataset_basepath, DATA_VALIDATION_SPLIT);
    dataset.validation = data_build(full_path, feature_size, num_classes),

    log_debug("Dataset train split: %d samples.", dataset.train->rows);
    log_debug("Dataset test split: %d samples.", dataset.test->rows);
    log_debug("Dataset validation split: %d samples.", dataset.validation->rows);

    if (dataset.train->rows == 0)
        log_warn("Training dataset is empty.");
    if (dataset.test->rows == 0)
        log_warn("Testing dataset is empty.");

    return dataset;
}

/**
 * Frees the memory allocated for the dataset.
 *
 * @param dataset The dataset to free.
 */
void free_dataset(Dataset dataset)
{
    log_debug("Freeing dataset splits.");
    free_data(dataset.train);
    free_data(dataset.test);
    free_data(dataset.validation);
}

/**
 * @brief Creates a new data object.
 *
 * @param feature_len The length of each input feature.
 * @param num_class The number of output classes.
 * @param rows The number of rows in the data object.
 * @return The pointer to the newly created data object.
 */
Data *new_data(const int feature_len, const int num_class, const int rows)
{
    log_debug("Creating new Data object with %d features, %d classes, and %d rows.", feature_len, num_class, rows);
    Data *data = malloc(sizeof(Data));
    data->input = new_matrix(rows, feature_len);
    data->target = new_matrix(rows, num_class);
    data->feature_len = feature_len;
    data->num_class = num_class;
    data->rows = rows;
    log_debug("Data object created at address %p.", (void *)data);
    return data;
}

/// TODO: adapt to contiguous memory allocation calling free once.
/**
 * @brief Frees a data object from the heap.
 *
 * @param data The data object to be freed.
 */
void free_data(Data *data)
{
    log_debug("Freeing Data object at address %p.", (void *)data);
    for (int row = 0; row < data->rows; row++)
    {
        free(data->input[row]);
        free(data->target[row]);
    }
    free(data->input);
    free(data->target);
    free(data);
}

/**
 * @brief Parses a string and extracts one row of inputs and outputs into the data object.
 *
 * @param data The data object to store the parsed values.
 * @param line The string containing the data values.
 * @param row The row index to store the values in.
 */
void parse_data(Data *data, char *line, const int row)
{
    const int cols = data->feature_len + data->num_class;
    for (int col = 0; col < cols; col++)
    {
        const double val = atof(strtok(col == 0 ? line : NULL, " "));
        if (col < data->feature_len)
            data->input[row][col] = val;
        else
            data->target[row][col - data->feature_len] = val;
    }
}

/**
 * @brief Randomly shuffles the rows of a data object.
 *
 * @param data The data object to be shuffled.
 */
void shuffle_data(Data *data)
{
    log_debug("Shuffling data object at address %p.", (void *)data);
    for (int a = 0; a < data->rows; a++)
    {
        const int b = get_random() % data->rows;
        double *ot = data->target[a];
        double *it = data->input[a];
        // Swap output.
        data->target[a] = data->target[b];
        data->target[b] = ot;
        // Swap input.
        data->input[a] = data->input[b];
        data->input[b] = it;
    }
}

/**
 * Creates a new batch of feedforward samples.
 *
 * @param batch_size The size of the batch.
 * @param sample_size The size of each sample.
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
void generate_samples(const Data *data, const int row, double *pos, double *neg)
{
    memcpy(pos, data->input[row], (data->feature_len - data->num_class) * sizeof(double));
    memcpy(neg, data->input[row], (data->feature_len - data->num_class) * sizeof(double));
    memcpy(&pos[data->feature_len - data->num_class], data->target[row], data->num_class * sizeof(double));
    // Set the negative sample's label to 0.0f
    memset(&neg[data->feature_len - data->num_class], 0, data->num_class * sizeof(double));
    // Find the label of the positive sample and store it in `one_pos`
    int one_pos = -1;
    for (int i = data->feature_len - data->num_class; i < data->feature_len; i++)
        if (pos[i] == 1.0f)
            // Store the index of the positive sample's label
            one_pos = i - (data->feature_len - data->num_class);
    // Generate a random label for the negative sample different from the positive sample's label
    int step = 1 + get_random() % (data->num_class - 1);
    int neg_label = (one_pos + step) % data->num_class;
    // Set the negative sample's label to 1.0f
    neg[(data->feature_len - data->num_class) + neg_label] = 1.0f;
}

/**
 * @brief Generates a batch of feedforward samples.
 *
 * @param data The data object.
 * @param row The row index of the data object.
 * @param batch The FFBatch object to store the generated samples.
 */
void generate_batch(const Data *data, const int batch_index, FFBatch batch)
{
    log_debug("Generating batch %d", batch_index);
    for (int i = 0; i < batch.size; i++)
    {
        const int index = (batch_index * batch.size + i) % data->rows;
        generate_samples(data, index, batch.pos[i], batch.neg[i]);
    }
}

/**
 * @brief Creates a new data object from a file.
 *
 * @param file_path The path to the file containing the data.
 * @param num_features The number of features in the dataset.
 * @param num_classes The number of classes in the dataset.
 *
 * @return The data object created from the file.
 */
Data *data_build(const char *file_path, const int num_features, const int num_classes)
{
    log_debug("Building data from %s", file_path);
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        log_error("Could not open %s", file_path);
        Data *empty_data = new_data(num_features, num_classes, 0);
        return empty_data;
    }
    const int rows = file_lines(file);
    Data *data = new_data(num_features, num_classes, rows);
    for (int row = 0; row < rows; row++)
    {
        char *line = read_line_from_file(file);
        parse_data(data, line, row);
        free(line);
    }
    fclose(file);
    log_debug("Built Data object at address: %p with %d samples", (void *)data, rows);
    return data;
}
