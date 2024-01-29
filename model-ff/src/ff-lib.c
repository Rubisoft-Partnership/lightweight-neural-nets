#include "ff-lib.h"

#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Buffer to store hidden activations and output activations.
#define H_BUFFER_SIZE 1000
float h_buffer[H_BUFFER_SIZE];
float o_buffer[H_BUFFER_SIZE];

// Function declarations.
static void ffbprop(const Tinn t, const float *const in_pos, const float *const in_neg,
                  const float rate, const float g_pos, const float g_neg, const float threshold);
static float fferr(const float g_pos, const float g_neg, const float threshold);
static float ffpderr(const float g_pos, const float g_neg, const float threshold);
static float goodness(const float *vec, const int size);
float fftrain(const Tinn t, const float *const pos, const float *const neg, float rate, const float threshold);

// From Tinn.c, but modified
void fprop(const Tinn t, const float* const in);

// From Tinn.c
static void wbrand(const Tinn t);
static float frand();



// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
float fftrain(const Tinn t, const float *const pos, const float *const neg, float rate, const float threshold)
{
    // Positive pass.
    fprop(t, pos);
    memcpy(h_buffer, t.h, t.nhid * sizeof(*t.h)); // copy activation and output
    memcpy(o_buffer, t.o, t.nops * sizeof(*t.o));
    float g_pos = goodness(t.o, t.nops);

    // Negative pass.
    fprop(t, neg);
    float g_neg = goodness(t.o, t.nops);

    ffbprop(t, pos, neg, rate, g_pos, g_neg, threshold);

    return fferr(g_pos, g_neg, threshold);
}

// Performs back propagation for the FF algorithm.
static void ffbprop(const Tinn t, const float *const in_pos, const float *const in_neg,
                  const float rate, const float g_pos, const float g_neg, const float threshold)
{
    for (int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        const float a = ffpderr(g_pos, g_neg, threshold);
        // Calculate total error change with respect to output.
        for (int j = 0; j < t.nops; j++)
        {
            const float b_pos = t.pdact(o_buffer[j]); // TODO: Change activation function and its derivative
            const float b_neg = t.pdact(t.o[j]);      // TODO: Change activation function and its derivative
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
    return log(1.0f + expf(-g_pos + threshold)) + log(1.0f + expf(g_neg - threshold));
}

// Returns partial derivative of error function.
static float ffpderr(const float g_pos, const float g_neg, const float threshold)
{
    return -expf(threshold) / (expf(g_pos) + expf(threshold)) + expf(threshold) / (expf(g_neg) + expf(threshold));
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




// Performs forward propagation.
void fprop(const Tinn t, const float* const in)
{
    // Calculate hidden layer neuron values.
    for(int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.h[i] = t.act(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for(int i = 0; i < t.nops; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.nhid; j++)
            sum += t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = t.act(sum + t.b[1]);
    }
}

// Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs
Tinn xtbuild(const int nips, const int nhid, const int nops, float (*act)(float), float (*pdact)(float))
{
    Tinn t;
    // Tinn only supports one hidden layer so there are two biases.
    t.nb = 2;
    t.nw = nhid * (nips + nops);
    t.w = (float*) calloc(t.nw, sizeof(*t.w));
    t.x = t.w + nhid * nips;
    t.b = (float*) calloc(t.nb, sizeof(*t.b));
    t.h = (float*) calloc(nhid, sizeof(*t.h));
    t.o = (float*) calloc(nops, sizeof(*t.o));
    t.nips = nips;
    t.nhid = nhid;
    t.nops = nops;
    t.act = act;
    t.pdact = pdact;
    wbrand(t);
    return t;
}

/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Below is the original Tinn code. It is in this file because of the way the Tinn library is structured.
///TODO: Fix imports and files structure.

// Loads a tinn from disk.
Tinn xtload(const char* const path)
{
    FILE* const file = fopen(path, "r");
    int nips = 0;
    int nhid = 0;
    int nops = 0;
    // Load header.
    fscanf(file, "%d %d %d\n", &nips, &nhid, &nops);
    // Build a new tinn.
    const Tinn t = xtbuild(nips, nhid, nops, relu, pdrelu);   ///TODO: relu hardcode is a quick fix, change this!
    // Load bias and weights.
    for(int i = 0; i < t.nb; i++) fscanf(file, "%f\n", &t.b[i]);
    for(int i = 0; i < t.nw; i++) fscanf(file, "%f\n", &t.w[i]);
    fclose(file);
    return t;
}



// Saves a tinn to disk.
void xtsave(const Tinn t, const char* const path)
{
    FILE* const file = fopen(path, "w");
    // Save header.
    fprintf(file, "%d %d %d\n", t.nips, t.nhid, t.nops);
    // Save biases and weights.
    for(int i = 0; i < t.nb; i++) fprintf(file, "%f\n", (double) t.b[i]);
    for(int i = 0; i < t.nw; i++) fprintf(file, "%f\n", (double) t.w[i]);
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
void xtprint(const float* arr, const int size)
{
    for(int i = 0; i < size; i++)
        printf("%f ", (double) arr[i]);
    printf("\n");
}


// Returns an output prediction given an input.
float* xtpredict(const Tinn t, const float* const in)
{
    fprop(t, in);
    return t.o;
}

// Randomizes tinn weights and biases.
static void wbrand(const Tinn t)
{
    for(int i = 0; i < t.nw; i++) t.w[i] = frand() - 0.5f;
    for(int i = 0; i < t.nb; i++) t.b[i] = frand() - 0.5f;
}

// Returns floating point random from 0.0 - 1.0.
static float frand()
{
    return rand() / (float) RAND_MAX;
}