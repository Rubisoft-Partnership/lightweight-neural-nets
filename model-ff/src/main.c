// #include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#include "ff-lib.h"

// Data object.
typedef struct
{
    // 2D floating point array of input.
    float **in;
    // 2D floating point array of target.
    float **tg;
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
    float *pos;
    // floating point array of FF negative sample <input, incorrect_label>
    float *neg;
} FFsamples;

// Returns the number of lines in a file.
static int lns(FILE *const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while ((ch = getc(file)) != EOF)
    {
        if (ch == '\n')
            lines++;
        pc = ch;
    }
    if (pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

// Reads a line from a file.
static char *readln(FILE *const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char *line = (char *)malloc((size) * sizeof(char));
    while ((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if (reads + 1 == size)
            line = (char *)realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// New 2D array of floats.
static float **new2d(const int rows, const int cols)
{
    float **row = (float **)malloc((rows) * sizeof(float *));
    for (int r = 0; r < rows; r++)
        row[r] = (float *)malloc((cols) * sizeof(float));
    return row;
}

// New data object.
static Data ndata(const int feature_len, const int num_class, const int rows)
{
    const Data data = {
        new2d(rows, feature_len), new2d(rows, num_class), feature_len, num_class, rows};
    return data;
}

// Gets one row of inputs and outputs from a string.
static void parse(const Data data, char *line, const int row)
{
    const int cols = data.feature_len + data.num_class;
    for (int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if (col < data.feature_len)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.feature_len] = val;
    }
}

// Frees a data object from the heap.
static void dfree(const Data d)
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
static void shuffle(const Data d)
{
    for (int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        float *ot = d.tg[a];
        float *it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

// Instantiates a new FFsamples object
static FFsamples new_samples(const int nips)
{
    FFsamples s = {
        (float *)malloc((nips) * sizeof(float)),
        (float *)malloc((nips) * sizeof(float))};
    return s;
}

// Generates a positive and a negative sample for the FF algorithm by embedding the one-hot encoded target in the input
static void generate_samples(const Data d, const int row, FFsamples s)
{
    memcpy(s.pos, d.in[row], (d.feature_len - d.num_class) * sizeof(float));
    memcpy(s.neg, d.in[row], (d.feature_len - d.num_class) * sizeof(float));
    memcpy(&s.pos[d.feature_len - d.num_class], d.tg[row], d.num_class * sizeof(float));
    memset(&s.neg[d.feature_len - d.num_class], 0, d.num_class * sizeof(float));
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

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.

// My understanding is that the dataset is fully loaded into memory for training.
// This is not ideal for large datasets, but it is fine for small datasets like the one int the README.
// TODO: decide if we want to load the entire dataset into memory or not and implement it
static Data build(const char *path, const int feature_len, const int num_class)
{
    FILE *file = fopen(path, "r");
    if (file == NULL)
    {
        printf("Could not open %s\n", path);
        printf("Get it:\n");
        printf("cd ../dataset/mnist; sh get_mnist_csv.sh .; python3 csv_to_tinn.py; cd -\n");
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(feature_len, num_class, rows);
    for (int row = 0; row < rows; row++)
    {
        char *line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

// Learns and predicts hand written digits with 98% accuracy.
int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
    const int nips = 784;
    const int nops = 10;
    const int layers_sizes[] = {784, 1500, 500, 10};
    const int layers_number = 4;
    // Hyper Parameters.
    // Learning rate is annealed and thus not constant.
    // It can be fine tuned along with the number of hidden layers.
    // Feel free to modify the anneal rate.
    // The number of iterations can be changed for stronger training.
    float rate = 0.5f;
    const float anneal = 0.99f;
    const int iterations = 12;
    const float threshold = 10.0f;

    open_log_file_with_timestamp("../logs", "ffnet");
    set_log_level(LOG_DEBUG);

    // Load the training set.
    const Data data = build("../../dataset/mnist/mnist_train.txt", nips, nops);
    // Train, baby, train.
    const FFNet ffnet = ffnetbuild(layers_sizes, layers_number, relu, pdrelu, threshold);
    FFsamples samples = new_samples(nips);
    for (int i = 0; i < iterations; i++)
    {
        log_info("Iteration %d", i);
        shuffle(data);
        float error = 0.0f;
        for (int j = 0; j < data.rows; j++)
        {
            log_debug("Sample %d", j);
            generate_samples(data, j, samples);

            error += fftrainnet(ffnet, samples.pos, samples.neg, rate);
            log_debug("Error %f", error);
        }
        printf("error %.12f :: learning rate %f\n",
               (double)error / data.rows,
               (double)rate);
        rate *= anneal;
    }

    log_info("Training done");

    // This is how you save the neural network to disk.

    /// TODO: implement saving and freeing of the neural network
    /*
    xtsave(ffnet, "saved.tinn");
    xtfree(ffnet);
    */
    // This is how you load the neural network from disk.
    // const Tinn loaded = xtload("saved.tinn");
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).

    // TODO: implement inference here
    const float *const in = data.in[0];
    const float *const tg = data.tg[0];
    const int pd = ffpredictnet(ffnet, in, nops, nips);
    // Prints target.
    xtprint(tg, data.num_class);
    // Prints prediction.
    printf("%d\n", pd);
    // xtprint(pd, data.num_class);
    // All done. Let's clean up.
    dfree(data);

    close_log_file();
    return 0;
}
