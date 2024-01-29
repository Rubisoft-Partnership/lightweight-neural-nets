// TODO: Add activation function and its derivative to Tinn

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




float fftrain(const Tinn t, const float *const pos, const float *const neg, float rate, const float threshold);

// New Tinn creation function that takes an activation function and its derivative as arguments
Tinn xtbuild(const int nips, const int nhid, const int nops, float (*act)(float), float (*pdact)(float));

// Activation function.
float relu(const float a);

float pdrelu(const float a);


/*
--------------------------------------------------------------------------------------------------------------------------
*/
// Tinn original functions

float *xtpredict(Tinn, const float *in);

float xttrain(Tinn, const float *in, const float *tg, float rate);

void xtsave(Tinn, const char *path);

Tinn xtload(const char *path);

void xtfree(Tinn);

void xtprint(const float *arr, const int size);
