/**
 * @file ff-utils.h
 * @brief Utility functions for a forward-forward neural networks.
 */
#pragma once


/**
 * @brief Generates a sample with the label embedded.
 * @param sample The output sample.
 * @param input The input values.
 * @param label The label to be embedded.
 * @param input_size The size of the input.
 * @param num_classes The number of classes.
 */
void embed_label(float *sample, const float *input, const int label, const int input_size, const int num_classes);

/**
 * @brief Normalizes a vector.
 * @param output The output vector.
 * @param size The size of the vector.
 */
void normalize_vector(float *output, int size);

/**
 * @brief Calculates the goodness of a vector.
 * @param vec The input vector.
 * @param size The size of the vector.
 * @return The goodness value.
 */
float goodness(const float *vec, const int size);
