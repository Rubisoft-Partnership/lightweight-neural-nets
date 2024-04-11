#include <stdlib.h>

#include <data/data.h>
#include <utils/utils.h>
#include <logging/logging.h>

// New data object.
Data new_data(const int feature_len, const int num_class, const int rows)
{
    const Data data = {
        new2d(rows, feature_len), new2d(rows, num_class), feature_len, num_class, rows};
    return data;
}

// Frees a data object from the heap.
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

// Gets one row of inputs and outputs from a string.
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

// Randomly shuffles a data object.
void shuffle_data(const Data d)
{
    for (int a = 0; a < d.rows; a++)
    {
        const int b = get_random() % d.rows;
        double *ot = d.target[a];
        double *it = d.input[a];
        // Swap output.
        d.target[a] = d.target[b];
        d.target[b] = ot;
        // Swap input.
        d.input[a] = d.input[b];
        d.input[b] = it;
    }
}

// Instantiates a new FFsamples object
FFsamples new_ff_samples(const int input_size)
{
    FFsamples s = {
        (double *)malloc((input_size) * sizeof(double)),
        (double *)malloc((input_size) * sizeof(double))};
    return s;
}

// Frees the memory of a FFsamples object
void free_ff_samples(FFsamples s)
{
    free(s.pos);
    free(s.neg);
}

// Generates a positive and a negative sample for the FF algorithm by embedding the one-hot encoded target in the input
void generate_samples(const Data d, const int row, FFsamples s)
{
    memcpy(s.pos, d.input[row], (d.feature_len - d.num_class) * sizeof(double));
    memcpy(s.neg, d.input[row], (d.feature_len - d.num_class) * sizeof(double));
    memcpy(&s.pos[d.feature_len - d.num_class], d.target[row], d.num_class * sizeof(double));
    memset(&s.neg[d.feature_len - d.num_class], 0, d.num_class * sizeof(double));
    // Set the positive sample's label to 0.0f
    int one_pos = -1;
    for (int i = d.feature_len - d.num_class; i < d.feature_len; i++)
        if (s.pos[i] == 1.0f)
            one_pos = i - (d.feature_len - d.num_class);
    // Generate a random label for the negative sample that is not the same as the positive sample's label
    int step = 1 + get_random() % (d.num_class - 1);
    int neg_label = (one_pos + step) % d.num_class;
    // Set the negative sample's label to 1.0f
    s.neg[(d.feature_len - d.num_class) + neg_label] = 1.0f;
}

///TODO: decide if we want to load the entire dataset into memory or not and implement it.
// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
Data data_build(void)
{
    log_debug("Building data from %s", DATA_DATASET_PATH);
    FILE *file = fopen(DATA_DATASET_PATH, "r");
    if (file == NULL)
    {
        printf("Could not open %s\n", DATA_DATASET_PATH);
        exit(1);
    }
    const int rows = lns(file);
    Data data = new_data(DATA_FEATURES, DATA_CLASSES, rows);
    for (int row = 0; row < rows; row++)
    {
        char *line = readln(file);
        parse_data(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}
