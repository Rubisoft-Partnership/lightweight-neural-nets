#include <ff-utils/ff-utils.h>

#include <string.h>
#include <math.h>

/**
 * Normalizes a vector.
 *
 * @param vec The vector to be normalized.
 * @param size The size of the vector.
 */
void normalize_vector(double *vec, int size)
{
    double norm = 0.0;
    for (int i = 0; i < size; i++)
        norm += vec[i] * vec[i];
    norm = sqrt(norm);
    for (int i = 0; i < size; i++)
        vec[i] /= norm;
}

/**
 * Calculates the goodness of a layer.
 *
 * @param vec The vector representing the layer.
 * @param size The size of the vector.
 * @return The goodness of the layer.
 */
double goodness(const double *vec, const int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += vec[i] * vec[i];
    return sum;
}

/**
 * Generates inputs for inference given input and label.
 *
 * @param sample The output sample vector.
 * @param input The input vector.
 * @param label The label for the input.
 * @param input_size The size of the input vector.
 * @param num_classes The number of classes.
 */
void embed_label(double *sample, const double *input, const int label, const int input_size, const int num_classes)
{
    memcpy(sample, input, input_size * sizeof(*input));
    memset(&sample[input_size - num_classes], 0, num_classes * sizeof(*sample));
    sample[input_size - num_classes + label] = 1.0;
}