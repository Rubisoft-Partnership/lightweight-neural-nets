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

// Activation functions.
double relu(const double a);
double pdrelu(const double a);

// Generates a FF cell.
Tinn new_ff_cell(const int nips, const int nops, double (*act)(double), double (*pdact)(double), const double threshold);
// Frees the memory of a FF cell.
void free_ff_cell(Tinn t);
// Trains a FF cell performing forward and backward pass with given a loss function.
double train_ff_cell(const Tinn t, const double *const pos, const double *const neg, double rate, const Loss loss_suite);
// Forward pass for a FF cell.
void fprop(const Tinn t, const double *const in);
// Generates a sample with the label embedded.
void embed_label(double *sample, const double *in, const int label, const int insize, const int num_classes);
// Normalizes a vector.
void normalize_vector(double *output, int size);
// Calculates the goodness of a vector.
double goodness(const double *vec, const int size);

