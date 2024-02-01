#pragma once

#include <stdio.h>

#define MAX_LAYERS_NUM 64


typedef struct
{
    // All the weights.
    float *w;
    // Hidden to output layer weights.
    float *x;
    // Biases.
    float *b;
    // Hidden layer.
    float *h;
    // Output layer.
    float *o;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    int nb;
    // Number of weights.
    int nw;
    // Number of inputs.
    int nips;
    // Number of hidden neurons.
    int nhid;
    // Number of outputs.
    int nops;
    // Hyperparameter for the FF algorithm.
    float threshold;
    // Activation function.
    float (*act)(const float);
    // Derivative of activation function.
    float (*pdact)(const float);
} Tinn; 


typedef struct
{
    Tinn hid_layers[MAX_LAYERS_NUM];
    int num_layers;
    int num_hid_layers;
}FFNet;


void open_log_file_with_timestamp(const char *logDir, const char *logPrefix);
void close_log_file();

typedef enum {
    LOG_DEBUG, // Detailed information, typically of interest only when diagnosing problems.
    LOG_INFO,  // Informational messages that highlight the progress of the application.
    LOG_WARN,  // Potentially harmful situations.
    LOG_ERROR  // Error events that might still allow the application to continue running.
} LogLevel;

void set_log_level(LogLevel level);


void log_debug(const char *format, ...);
void log_info(const char *format, ...);
void log_warn(const char *format, ...);
void log_error(const char *format, ...);



float fftrainnet(const FFNet ffnet, const float *const pos, const float *const neg, float rate);
FFNet ffnetbuild(const int *layer_sizes, int num_layers, float (*act)(float), float (*pdact)(float), const float treshold);
int ffpredictnet(const FFNet ffnet, const float *in, int num_classes, int insize);




// Activation function.

float relu(const float a);
float pdrelu(const float a);

float sigmoid(const float a);
float pdsigmoid(const float a);


/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Tinn original functions


void xtprint(const float *arr, const int size);
