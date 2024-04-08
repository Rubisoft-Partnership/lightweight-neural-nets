#pragma once

#include <stdio.h>
#include <adam/adam.h>
#include <losses/losses.h>

// Size of buffer to store hidden activations and output activations.
#define H_BUFFER_SIZE 1024

#define MAX_CLASSES 16

typedef struct
{
    // All the weights.
    double *w;
    // Biases.
    double b;
    // Output layer.
    double *o;
    // Number of biases.
    int nb;
    // Number of weights.
    int nw;
    // Number of inputs.
    int nips;
    // Number of outputs.
    int nops;
    // Hyperparameter for the FF algorithm.
    double threshold;
    // Activation function.
    double (*act)(const double);
    // Derivative of activation function.
    double (*pdact)(const double);
    // Adam optimizer.
    Adam adam;
} Tinn;

// Activation function.

double relu(const double a);
double pdrelu(const double a);

double sigmoid(const double a);
double pdsigmoid(const double a);
double fftrain(const Tinn t, const double *const pos, const double *const neg, double rate, const Loss loss_suite);
Tinn xtbuild(const int nips, const int nops, double (*act)(double), double (*pdact)(double), const double threshold);
void xtfree(Tinn t);
void embed_label(double *sample, const double *in, const int label, const int insize, const int num_classes);
void normalize_vector(double *output, int size);
double goodness(const double *vec, const int size);
void fprop(const Tinn t, const double *const in);


/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Tinn original functions

void xtprint(const double *arr, const int size);
