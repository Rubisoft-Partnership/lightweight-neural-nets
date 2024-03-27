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

double weight_update(Adam adam, const double gradient, const int index)
{
    // Increment time step
    adam.t++;

    // Update the Adam optimizer
    adam.m[index] = adam.beta1 * adam.m[index] + (1 - adam.beta1) * gradient;
    adam.v[index] = adam.beta2 * adam.v[index] + (1 - adam.beta2) * gradient * gradient;

    // Bias correction
    const double m_hat = adam.m[index] / (1 - pow(adam.beta1, adam.t));
    const double v_hat = adam.v[index] / (1 - pow(adam.beta2, adam.t));

    // Weight update using Adam optimizer
    return m_hat / (sqrt(v_hat) + 1e-8);
}
