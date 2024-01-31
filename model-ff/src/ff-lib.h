#pragma once

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
