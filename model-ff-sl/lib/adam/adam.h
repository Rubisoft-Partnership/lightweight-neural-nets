#pragma once

#include <string.h>
#include <stdlib.h>

// Adam parameters
typedef struct
{
    // Adam hyperparameters
    double beta1;
    double beta2;

    // m and v zero-initializalized vectors
    // First moment estimate
    double *m;
    // Second moment estimate
    double *v;

    // Time step
    int t;
} Adam;

// Adam constructor
Adam adam_create(double beta1, double beta2, int size);

// Adam destructor
void adam_free(Adam adam);

// Adam weight update
double  adam_weight_update(Adam adam, double gradient, int index);

