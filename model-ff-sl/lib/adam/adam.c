/**
 * @file adam.c
 * @brief Implementation of the Adam optimizer.
 */

#include <adam/adam.h>
#include <math.h>

/**
 * @brief Creates an Adam optimizer with the given beta1, beta2, and size.
 * 
 * @param beta1 The exponential decay rate for the first moment estimates.
 * @param beta2 The exponential decay rate for the second moment estimates.
 * @param size The size of the weight vector.
 * @return The created Adam optimizer.
 */
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

/**
 * @brief Frees the memory allocated for the Adam optimizer.
 * 
 * @param adam The Adam optimizer to free.
 */
void adam_free(Adam adam)
{
    free(adam.m);
    free(adam.v);
}

/**
 * @brief Updates the weight using the Adam optimizer.
 * 
 * @param adam The Adam optimizer.
 * @param gradient The gradient of the weight.
 * @param index The index of the weight.
 * @return The updated weight.
 */
double adam_weight_update(Adam adam, const double gradient, const int index)
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
