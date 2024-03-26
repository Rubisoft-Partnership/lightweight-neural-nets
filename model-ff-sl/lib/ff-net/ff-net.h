#pragma once

#include <stdio.h>

#include <ff-cell/ff-cell.h>

#define MAX_LAYERS_NUM 64


typedef struct
{
    Tinn layers[MAX_LAYERS_NUM];
    int num_cells;
} FFNet;

double fftrainnet(const FFNet ffnet, const double *const pos, const double *const neg, double rate);
FFNet ffnetbuild(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double treshold);
void ffnetfree(FFNet ffnet);
int ffpredictnet(const FFNet ffnet, const double *in, const int num_classes, const int insize);

// Activation function.

double relu(const double a);
double pdrelu(const double a);

double sigmoid(const double a);
double pdsigmoid(const double a);

