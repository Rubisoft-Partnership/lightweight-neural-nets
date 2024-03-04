#pragma once

#include <stdio.h>

// Size of buffer to store hidden activations and output activations.
#define H_BUFFER_SIZE 1024

#define MAX_CLASSES 16

typedef struct
{
    // All the weights.
    double *w;
    // Hidden to output layer weights.
    double *x;
    // Biases.
    double *b;
    // Hidden layer.
    double *h;
    // Output layer.
    double *o;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    int nb;
    // Number of weights.
    int nw;
    // Number of inputs.
    int nips;
    // Number of hidden neurons.
    int nhid;
    // Number of outputs.
    int nops;
    // Hyperparameter for the FF algorithm.
    double threshold;
    // Activation function.
    double (*act)(const double);
    // Derivative of activation function.
    double (*pdact)(const double);
} Tinn;

// Activation function.

double relu(const double a);
double pdrelu(const double a);

double sigmoid(const double a);
double pdsigmoid(const double a);
double fftrain(const Tinn t, const double *const pos, const double *const neg, double rate);
Tinn xtbuild(const int nips, const int nhid, const int nops, double (*act)(double), double (*pdact)(double), const double threshold);
void embed_label(double *sample, const double *in, int label, int insize, int num_classes);
void normalize_vector(double *output, int size);
double goodness(const double *vec, const int size);

// From Tinn.c, but modified
void fprop(const Tinn t, const double *const in);
/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Tinn original functions

void xtprint(const double *arr, const int size);
