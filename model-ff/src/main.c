#include "Tinn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

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
    int label_len;
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
static Data ndata(const int feature_len, const int label_len, const int rows)
{
    const Data data = {
        new2d(rows, feature_len), new2d(rows, label_len), feature_len, label_len, rows};
    return data;
}

// Gets one row of inputs and outputs from a string.
static void parse(const Data data, char *line, const int row)
{
    const int cols = data.feature_len + data.label_len;
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

// Generates a positive and a negative sample for the FF algorithm by appending the one-hot encoded target to the input
static void generate_samples(const Data d, const int row, FFsamples s)
{
    for (int col = 0; col < d.feature_len; col++)
    {
        s.pos[col] = d.in[row][col];
        s.neg[col] = d.in[row][col];
    }
    int one_pos;
    for (int col = d.feature_len; col < d.feature_len + d.label_len; col++)
    {
        if (d.tg[row][col - d.feature_len] == 1.0f)
        {
            one_pos = col;
        }
        s.pos[col] = d.tg[row][col - d.feature_len];
        s.neg[col] = d.tg[row][col - d.feature_len];
    }
    // Set the positive sample's label to 0.0f
    s.neg[one_pos] = 0.0f;
    // Generate a random label for the negative sample that is not the same as the positive sample's label
    int neg_label = rand() % (d.label_len - 1);
    if (neg_label >= one_pos)
    {
        neg_label++;
    }
    // Set the negative sample's label to 1.0f
    s.neg[d.feature_len + neg_label] = 1.0f;
}

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.

// My understanding is that the dataset is fully loaded into memory for training.
// This is not ideal for large datasets, but it is fine for small datasets like the one int the README.
// TODO: decide if we want to load the entire dataset into memory or not and implement it
static Data build(const char *path, const int feature_len, const int label_len)
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
    Data data = ndata(feature_len, label_len, rows);
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
    const int nips = 784 + 10;
    const int nops = 10;
    // Hyper Parameters.
    // Learning rate is annealed and thus not constant.
    // It can be fine tuned along with the number of hidden layers.
    // Feel free to modify the anneal rate.
    // The number of iterations can be changed for stronger training.
    float rate = 1.0f;
    const int nhid = 28;
    const float anneal = 0.99f;
    const int iterations = 128;
    // Load the training set.
    const Data data = build("../../dataset/mnist/mnist_train.txt", nips - nops, nops);
    // Train, baby, train.
    const Tinn tinn = xtbuild(nips, nhid, nops);
    FFsamples samples = new_samples(nips);
    for (int i = 0; i < iterations; i++)
    {
        shuffle(data);
        float error = 0.0f;
        for (int j = 0; j < data.rows; j++)
        {
            generate_samples(data, j, samples);

            // TODO: edit `xttrain` to take in the positive and negative samples in the form of <input, label>
            //  error += xttrain(tinn, in_pos, tg_pos, rate);
        }
        printf("error %.12f :: learning rate %f\n",
               (double)error / data.rows,
               (double)rate);
        rate *= anneal;
    }
    // This is how you save the neural network to disk.
    xtsave(tinn, "saved.tinn");
    xtfree(tinn);
    // This is how you load the neural network from disk.
    const Tinn loaded = xtload("saved.tinn");
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).

    // TODO: implement inference here
    const float *const in = data.in[0];
    const float *const tg = data.tg[0];
    const float *const pd = xtpredict(loaded, in);
    // Prints target.
    xtprint(tg, data.label_len);
    // Prints prediction.
    xtprint(pd, data.label_len);
    // All done. Let's clean up.
    xtfree(loaded);
    dfree(data);
    return 0;
}
