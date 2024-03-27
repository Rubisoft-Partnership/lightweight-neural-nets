#include <adam/adam.h>


Adam adam_create(const double beta1, const double beta2, const int size)
{
    Adam adam;
    adam.beta1 = beta1;
    adam.beta2 = beta2;

    // Initialize time step to 1
    adam.t = 1;
    
    // Allocate memory for m and v
    adam.m = malloc(size * sizeof(double));
    adam.v = malloc(size * sizeof(double));

    // Initialize m and v to 0
    memset(adam.m, 0, size * sizeof(double));
    memset(adam.v, 0, size * sizeof(double));

    return adam;
}

void adam_free(Adam adam)
{
    free(adam.m);
    free(adam.v);
}
