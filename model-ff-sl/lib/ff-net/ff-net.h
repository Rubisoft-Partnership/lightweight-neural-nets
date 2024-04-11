#pragma once

#include <stdio.h>

#include <ff-cell/ff-cell.h>
#include <losses/losses.h>

#define MAX_LAYERS_NUM 64


typedef struct
{
    Tinn layers[MAX_LAYERS_NUM];
    int num_cells;
    Loss loss_suite;
} FFNet;

double train_ff_net(const FFNet ffnet, const double *const pos, const double *const neg, double rate);
FFNet new_ff_net(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double treshold, Loss loss_suite);
void free_ff_net(FFNet ffnet);
int predict_ff_net(const FFNet ffnet, const double *in, const int num_classes, const int insize);

// Activation function.

double relu(const double a);
double pdrelu(const double a);


