#include "ff-lib.h"

#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Buffer to store hidden activations and output activations.
#define H_BUFFER_SIZE 1024
float h_buffer[H_BUFFER_SIZE]; // activations buffer
float o_buffer[H_BUFFER_SIZE]; // outputs buffer

#define MAX_CLASSES 16

// Function declarations.
static void ffbprop(const Tinn t, const float *const in_pos, const float *const in_neg,
                    const float rate, const float g_pos, const float g_neg);
static float fferr(const float g_pos, const float g_neg, const float threshold);
static float ffpderr(const float g_pos, const float g_neg, const float threshold);
static float goodness(const float *vec, const int size);
float fftrain(const Tinn t, const float *const pos, const float *const neg, float rate);
Tinn xtbuild(const int nips, const int nhid, const int nops, float (*act)(float), float (*pdact)(float), const float threshold);
void embed_label(float *sample, const float *in, int label, int insize, int num_classes);
void normalize_vector(float *output, int size);

// From Tinn.c, but modified
void fprop(const Tinn t, const float *const in);

// From Tinn.c
static void wbrand(const Tinn t);
static float frand();

// Builds a FFNet by creating multiple Tinn objects. layer_sizes includes the number of inputs, hidden neurons, and outputs units.
FFNet ffnetbuild(const int *layer_sizes, int num_layers, float (*act)(float), float (*pdact)(float), const float treshold)
{
    FFNet ffnet;
    ffnet.num_layers = num_layers;
    ffnet.num_hid_layers = num_layers - 2;
    for (int i = 1; i < num_layers - 1; i++)
    {
        ffnet.hid_layers[i - 1] = xtbuild(layer_sizes[i - 1], layer_sizes[i], layer_sizes[i + 1], act, pdact, treshold);
    }

    return ffnet;
}

float fftrainnet(const FFNet ffnet, const float *const pos, const float *const neg, float rate)
{
    // printf("Training FFNet...\n");
    float error = 0.0f;
    // Feed first layer manually.
    error += fftrain(ffnet.hid_layers[0], pos, neg, rate);
    // Feed the rest of the layers.
    for (int i = 1; i < ffnet.num_hid_layers; i++)
    {
        error += fftrain(ffnet.hid_layers[i], o_buffer, ffnet.hid_layers[i - 1].o, rate);
    }
    // printf("error: %f\n", error);
    return error;
}

// Inference function for FFNet.
int ffpredictnet(const FFNet ffnet, const float *in, int num_classes, int insize)
{
    float *netinput = (float *)malloc((insize) * sizeof(float));
    float goodnesses[MAX_CLASSES];
    for (int label = 0; label < num_classes; label++)
    {
        embed_label(netinput, in, label, insize, num_classes);
        fprop(ffnet.hid_layers[0], in);
        normalize_vector(ffnet.hid_layers[0].o, ffnet.hid_layers[0].nops);
        for (int i = 1; i < ffnet.num_hid_layers; i++)
        {
            fprop(ffnet.hid_layers[i], ffnet.hid_layers[i - 1].o);
            normalize_vector(ffnet.hid_layers[i].o, ffnet.hid_layers[i].nops);
            goodnesses[label] += goodness(ffnet.hid_layers[i].o, ffnet.hid_layers[i].nops);
        }
    }

    free(netinput);

    int max_goodness_index = 0;
    for (int i = 1; i < num_classes; i++)
    {
        if (goodnesses[i] > goodnesses[max_goodness_index])
            max_goodness_index = i;
    }
    return max_goodness_index;
}

// Generates inputs for inference given input and label
void embed_label(float *sample, const float *in, int label, int insize, int num_classes)
{
    memcpy(sample, in, insize * sizeof(*in));
    memset(&sample[insize - num_classes], 0, num_classes * sizeof(*sample));
    sample[insize - label] = 1.0f;
}

// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
float fftrain(const Tinn t, const float *const pos, const float *const neg, float rate)
{
    // Positive pass.
    fprop(t, pos);
    memcpy(h_buffer, t.h, t.nhid * sizeof(*t.h)); // copy activation and output
    memcpy(o_buffer, t.o, t.nops * sizeof(*t.o));
    float g_pos = goodness(t.o, t.nops);

    // Negative pass.
    fprop(t, neg);
    float g_neg = goodness(t.o, t.nops);

    // Peforms gradient descent.
    ffbprop(t, pos, neg, rate, g_pos, g_neg);

    // Normalize the output of the layer
    normalize_vector(t.o, t.nops);
    normalize_vector(o_buffer, t.nops);

    // printf("g_pos: %f, g_neg: %f, err: %f\n", g_pos, g_neg, fferr(g_pos, g_neg, t.threshold));
    return fferr(g_pos, g_neg, t.threshold);
}

void normalize_vector(float *output, int size)
{
    float norm = 0.0f;
    for (int i = 0; i < size; i++)
        norm += output[i] * output[i];
    norm = sqrt(norm);
    for (int i = 0; i < size; i++)
        output[i] /= norm;
}

// Performs back propagation for the FF algorithm.
static void ffbprop(const Tinn t, const float *const in_pos, const float *const in_neg,
                    const float rate, const float g_pos, const float g_neg)
{
    const float a = ffpderr(g_pos, g_neg, t.threshold);
    printf("a: %f\n", a);
    for (int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        // Calculate total error change with respect to output.
        for (int j = 0; j < t.nops; j++)
        {
            const float b_pos = t.pdact(o_buffer[j]);
            const float b_neg = t.pdact(t.o[j]);
            sum += a * (b_pos + b_neg) * t.x[j * t.nhid + i];
            // Correct weights in hidden to output layer.
            t.x[j * t.nhid + i] -= rate * a * (b_pos * h_buffer[i] + b_neg * t.h[i]);
        }
        // Correct weights in input to hidden layer.
        for (int j = 0; j < t.nips; j++)
            t.w[i * t.nips + j] -= rate * sum * (t.pdact(t.h[i]) * in_neg[j] + t.pdact(h_buffer[i]) * in_pos[j]);
    }
}

// Computes error using the FFLoss function.
static float fferr(const float g_pos, const float g_neg, const float threshold)
{

    float pos_exponent = -g_pos + threshold;
    float neg_exponent = g_neg - threshold;
    float first_term = logf(1 + expf(-abs(pos_exponent))) + pos_exponent > 0.0 ? pos_exponent : 0.0;
    float second_term = logf(1 + expf(-abs(neg_exponent))) + neg_exponent > 0.0 ? neg_exponent : 0.0;

    printf("g_pos: %f, g_neg: %f, err: %f\n", g_pos, g_neg, first_term + second_term);

    return first_term + second_term;

    // equivalent to:
    // return logf(1.0f + expf(-g_pos + threshold)) + logf(1.0f + expf(g_neg - threshold));
}

static float stable_sigmoid(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + expf(-x) + 1e-4);
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x + 1e-4);
    }
}

// Returns partial derivative of error function.
static float ffpderr(const float g_pos, const float g_neg, const float threshold)
{
    float sigmoid_g_pos = stable_sigmoid(threshold - g_pos);
    float sigmoid_g_neg = stable_sigmoid(threshold - g_neg);

    return -sigmoid_g_pos + sigmoid_g_neg + 1e-4;
    // return -expf(threshold) / (expf(g_pos) + expf(threshold)) + expf(threshold) / (expf(g_neg) + expf(threshold));
}

// Returns the goodness of a layer.
float goodness(const float *vec, const int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        sum += vec[i] * vec[i];
    }
    return sum;
}

// ReLU activation function.
float relu(const float a)
{
    return a > 0.0f ? a : 0.0f;
}

// ReLU derivative.
float pdrelu(const float a)
{
    return a > 0.0f ? 1.0f : 0.0f;
}

// Sigmoid activation function.
float sigmoid(const float a)
{
    return 1.0f / (1.0f + expf(-a));
}

// Sigmoid derivative.
float pdsigmoid(const float a)
{
    return a * (1.0f - a);
}

// Performs forward propagation.
void fprop(const Tinn t, const float *const in)
{
    // Calculate hidden layer neuron values.
    for (int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.h[i] = t.act(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for (int i = 0; i < t.nops; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < t.nhid; j++)
            sum += t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = t.act(sum + t.b[1]);
    }
}

// Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
Tinn xtbuild(const int nips, const int nhid, const int nops, float (*act)(float), float (*pdact)(float), const float threshold)
{
    Tinn t;
    // Tinn only supports one hidden layer so there are two biases.
    t.nb = 2;
    t.nw = nhid * (nips + nops);               // total number of weights
    t.w = (float *)calloc(t.nw, sizeof(*t.w)); // weights (both [intput to hidden] and [hidden to output])
    t.x = t.w + nhid * nips;
    t.b = (float *)calloc(t.nb, sizeof(*t.b)); // biases
    t.h = (float *)calloc(nhid, sizeof(*t.h)); // hidden neurons
    t.o = (float *)calloc(nops, sizeof(*t.o)); // output neurons
    t.nips = nips;
    t.nhid = nhid;
    t.nops = nops;
    t.act = act;
    t.pdact = pdact;
    t.threshold = threshold;
    wbrand(t);
    return t;
}

/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Below is the original Tinn code. It is in this file because of the way the Tinn library is structured.
/// TODO: Fix imports and files structure.

// Loads a tinn from disk.
Tinn xtload(const char *const path)
{
    FILE *const file = fopen(path, "r");
    int nips = 0;
    int nhid = 0;
    int nops = 0;
    // Load header.
    fscanf(file, "%d %d %d\n", &nips, &nhid, &nops);
    // Build a new tinn.
    const Tinn t = xtbuild(nips, nhid, nops, sigmoid, pdsigmoid, 0.5); /// TODO: relu and treshold hardcode is a quick fix, change this!
    // Load bias and weights.
    for (int i = 0; i < t.nb; i++)
        fscanf(file, "%f\n", &t.b[i]);
    for (int i = 0; i < t.nw; i++)
        fscanf(file, "%f\n", &t.w[i]);
    fclose(file);
    return t;
}

// Saves a tinn to disk.
void xtsave(const Tinn t, const char *const path)
{
    FILE *const file = fopen(path, "w");
    // Save header.
    fprintf(file, "%d %d %d\n", t.nips, t.nhid, t.nops);
    // Save biases and weights.
    for (int i = 0; i < t.nb; i++)
        fprintf(file, "%f\n", (double)t.b[i]);
    for (int i = 0; i < t.nw; i++)
        fprintf(file, "%f\n", (double)t.w[i]);
    fclose(file);
}

// Frees object from heap.
void xtfree(const Tinn t)
{
    free(t.w);
    free(t.b);
    free(t.h);
    free(t.o);
}

// Prints an array of floats. Useful for printing predictions.
void xtprint(const float *arr, const int size)
{
    for (int i = 0; i < size; i++)
        printf("%f ", (double)arr[i]);
    printf("\n");
}

// Returns an output prediction given an input.
float *xtpredict(const Tinn t, const float *const in)
{
    fprop(t, in);
    return t.o;
}

// Randomizes tinn weights and biases.
static void wbrand(const Tinn t)
{
    for (int i = 0; i < t.nw; i++)
        t.w[i] = frand() - 0.5f;
    for (int i = 0; i < t.nb; i++)
        t.b[i] = frand() - 0.5f;
}

// Returns floating point random from 0.0 - 1.0.
static float frand()
{
    return rand() / (float)RAND_MAX;
}