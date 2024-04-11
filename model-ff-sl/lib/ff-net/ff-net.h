#pragma once

#include <ff-cell/ff-cell.h>
#include <losses/losses.h>

#define MAX_LAYERS_NUM 64

// FFNet struct that contains multiple Tinn objects.
typedef struct
{
    Tinn layers[MAX_LAYERS_NUM];
    int num_cells;
    Loss loss_suite;
} FFNet;

// Builds a FFNet by creating multiple Tinn objects.
FFNet new_ff_net(const int *layer_sizes, int num_layers, double (*act)(double), double (*pdact)(double), const double treshold, Loss loss_suite);
// Frees the memory of a FFNet.
void free_ff_net(FFNet ffnet);
// Trains a FFNet by training each cell.
double train_ff_net(const FFNet ffnet, const double *const pos, const double *const neg, double rate);
// Inference function for FFNet.
int predict_ff_net(const FFNet ffnet, const double *in, const int num_classes, const int insize);


