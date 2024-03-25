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
    double *m;
    double *v;
} Adam;

// Adam constructor
Adam adam_create(double beta1, double beta2, int size);

