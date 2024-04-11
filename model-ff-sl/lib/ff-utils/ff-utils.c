#include <ff-utils/ff-utils.h>

#include <string.h>
#include <math.h>

// Normalizes a vector.
void normalize_vector(double *vec, int size)
{
    double norm = 0.0;
    for (int i = 0; i < size; i++)
        norm += vec[i] * vec[i];
    norm = sqrt(norm);
    for (int i = 0; i < size; i++)
        vec[i] /= norm;
}

// Returns the goodness of a layer.
double goodness(const double *vec, const int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += vec[i] * vec[i];
    return sum;
}

// Generates inputs for inference given input and label
void embed_label(double *sample, const double *in, const int label, const int in_size, const int num_classes)
{
    memcpy(sample, in, in_size * sizeof(*in));
    memset(&sample[in_size - num_classes], 0, num_classes * sizeof(*sample));
    sample[in_size - num_classes + label] = 1.0;
}