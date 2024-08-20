#ifndef MODEL_FF_H
#define MODEL_FF_H

#include <vector>
#include <random>
#include <metrics.hpp>

extern "C"
{
#include <ff-net/ff-net.h>
#include <data/data.h>
}


class ModelFF
{
public:
    // Destructor
    virtual ~ModelFF() {}

    // Initialize the model with necessary parameters or configurations
    void build();

    // Train the model for a given number of epochs
    void train(const int &epochs, const int &batch_size, const double &learning_rate);

    // Evaluate the model's performance with the given test data and labels
    // metrics::Metrics evaluate();


private:
    FFNet *ffnet;

    int max_units;
    int input_size;
};

#endif // MODEL_FF_H
