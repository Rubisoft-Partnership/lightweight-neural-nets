#pragma once

// Generates inputs for inference given input and label
void embed_label(double *sample, const double *in, const int label, const int in_size, const int num_classes);

// Returns the goodness of a layer.
double goodness(const double *vec, const int size);

// Normalizes a vector.
void normalize_vector(double *vec, int size);
