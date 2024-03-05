#include <stdlib.h>

#include <data/data.h>
#include <utils/utils.h>

// New data object.
Data ndata(const int feature_len, const int num_class, const int rows)
{
    const Data data = {
        new2d(rows, feature_len), new2d(rows, num_class), feature_len, num_class, rows};
    return data;
}

// Gets one row of inputs and outputs from a string.
void parse(const Data data, char *line, const int row)
{
    const int cols = data.feature_len + data.num_class;
    for (int col = 0; col < cols; col++)
    {
        const double val = atof(strtok(col == 0 ? line : NULL, " "));
        if (col < data.feature_len)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.feature_len] = val;
    }
}

// Frees a data object from the heap.
void dfree(const Data d)
{
    for (int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

// Randomly shuffles a data object.
void shuffle(const Data d)
{
    for (int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        double *ot = d.tg[a];
        double *it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

// Instantiates a new FFsamples object
FFsamples new_samples(const int nips)
{
    FFsamples s = {
        (double *)malloc((nips) * sizeof(double)),
        (double *)malloc((nips) * sizeof(double))};
    return s;
}

// Generates a positive and a negative sample for the FF algorithm by embedding the one-hot encoded target in the input
void generate_samples(const Data d, const int row, FFsamples s)
{
    memcpy(s.pos, d.in[row], (d.feature_len - d.num_class) * sizeof(double));
    memcpy(s.neg, d.in[row], (d.feature_len - d.num_class) * sizeof(double));
    memcpy(&s.pos[d.feature_len - d.num_class], d.tg[row], d.num_class * sizeof(double));
    memset(&s.neg[d.feature_len - d.num_class], 0, d.num_class * sizeof(double));
    // Set the positive sample's label to 0.0f
    int one_pos;
    for (int i = d.feature_len - d.num_class; i < d.feature_len; i++)
        if (s.pos[i] == 1.0f)
            one_pos = i - (d.feature_len - d.num_class);
    // Generate a random label for the negative sample that is not the same as the positive sample's label
    int step = 1 + rand() % (d.num_class - 1);
    int neg_label = (one_pos + step) % d.num_class;
    // Set the negative sample's label to 1.0f
    s.neg[(d.feature_len - d.num_class) + neg_label] = 1.0f;
}

// TODO: decide if we want to load the entire dataset into memory or not and implement it.
// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
Data build(void)
{
    log_debug("Building data from %s", DATA_DATASET_PATH);
    FILE *file = fopen(DATA_DATASET_PATH, "r");
    if (file == NULL)
    {
        printf("Could not open %s\n", DATA_DATASET_PATH);
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(DATA_FEATURES, DATA_CLASSES, rows);
    for (int row = 0; row < rows; row++)
    {
        char *line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}
