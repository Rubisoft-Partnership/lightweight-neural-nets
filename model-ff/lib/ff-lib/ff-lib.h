#pragma once

#include <stdio.h>

#define MAX_LAYERS_NUM 64

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

typedef struct
{
    Tinn hid_layers[MAX_LAYERS_NUM];
    int num_layers;
    int num_hid_layers;
} FFNet;

double fftrainnet(const FFNet ffnet, const double *const pos, const double *const neg, double rate);
FFNet ffnetbuild(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double treshold);
int ffpredictnet(const FFNet ffnet, const double *in, int num_classes, int insize);

// Activation function.

double relu(const double a);
double pdrelu(const double a);

double sigmoid(const double a);
double pdsigmoid(const double a);

/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Tinn original functions

void xtprint(const double *arr, const int size);
