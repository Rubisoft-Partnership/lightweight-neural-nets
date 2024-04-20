/**
 * @file data.c
 * @brief This file contains the implementation of data-related functions.
 */

#include <stdlib.h>

#include <data/data.h>
#include <utils/utils.h>
#include <logging/logging.h>

/**
 * Splits the dataset into training, testing, and validation sets.
 *
 * @return The dataset containing the split sets.
 */
Dataset dataset_split(void)
{
    Dataset dataset;
    dataset.train = data_build(DATA_TRAIN_PATH);
    dataset.test = data_build(DATA_TEST_PATH);
    dataset.validation = data_build(DATA_VALIDATION_PATH);

    log_debug("Dataset train split: %d samples.", dataset.train->rows);
    log_debug("Dataset test split: %d samples.", dataset.test->rows);
    log_debug("Dataset validation split: %d samples.", dataset.validation->rows);

    if (dataset.train->rows == 0 || dataset.test->rows == 0)
    {
        log_error("Train and test splits cannot be empty. Exiting.");
        free_dataset(dataset);
        exit(EXIT_FAILURE);
    }
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
 * @brief Creates a new FFsamples object.
 * 
 * @param input_size The size of the input array.
 * @return The newly created FFsamples object.
 */
FFsamples new_ff_samples(const int input_size)
{
    log_debug("Creating new FFsamples object with input size %d.", input_size);
    FFsamples samples = {
        (double *)malloc((input_size) * sizeof(double)),
        (double *)malloc((input_size) * sizeof(double))};
    log_debug("FFsamples object created at address %p.", (void *)&samples);
    return samples;
}

/**
 * @brief Frees the memory of a FFsamples object.
 * 
 * @param samples The FFsamples object to be freed.
 */
void free_ff_samples(FFsamples samples)
{
    log_debug("Freeing FFsamples object at address %p.", (void *)&samples);
    free(samples.pos);
    free(samples.neg);
}

/**
 * @brief Generates a positive and a negative sample for the FF algorithm by embedding the one-hot encoded target in the input.
 * 
 * @param data The data object containing the input and target values.
 * @param row The row index to generate the samples from.
 * @param samples The FFsamples object to store the generated samples.
 */
void generate_samples(const Data *data, const int row, FFsamples samples)
{
    memcpy(samples.pos, data->input[row], (data->feature_len - data->num_class) * sizeof(double));
    memcpy(samples.neg, data->input[row], (data->feature_len - data->num_class) * sizeof(double));
    memcpy(&samples.pos[data->feature_len - data->num_class], data->target[row], data->num_class * sizeof(double));
    memset(&samples.neg[data->feature_len - data->num_class], 0, data->num_class * sizeof(double));
    // Set the positive sample's label to 0.0f
    int one_pos = -1;
    for (int i = data->feature_len - data->num_class; i < data->feature_len; i++)
        if (samples.pos[i] == 1.0f)
            one_pos = i - (data->feature_len - data->num_class);
    // Generate a random label for the negative sample that is not the same as the positive sample's label
    int step = 1 + get_random() % (data->num_class - 1);
    int neg_label = (one_pos + step) % data->num_class;
    // Set the negative sample's label to 1.0f
    samples.neg[(data->feature_len - data->num_class) + neg_label] = 1.0f;
}

/**
 * @brief Builds a data object by parsing a file and extracting the inputs and outputs for the neural network.
 * 
 * @param file_path The path of the file to read from.
 * @return The built data object.
 */
Data *data_build(const char *file_path)
{
    log_debug("Building data from %s", file_path);
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        log_error("Could not open %s\n", file_path);
        Data *empty_data = new_data(DATA_FEATURES, DATA_CLASSES, 0);
        return empty_data;
    }
    const int rows = file_lines(file);
    Data *data = new_data(DATA_FEATURES, DATA_CLASSES, rows);
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
