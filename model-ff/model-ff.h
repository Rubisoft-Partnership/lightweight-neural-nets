#ifndef MODEL_FF_H
#define MODEL_FF_H

#include "../framework/lib/model/model.h"
#include <vector>
#include <random>
#include <metrics.hpp>

extern "C"
{
#include <ff-net/ff-net.h>
#include <data/data.h>
}

class ModelFF : public Model
{
public:
    // Destructor
    virtual ~ModelFF() {}

    // Initialize the model with necessary parameters or configurations
    void build(const std::vector<int> &units, const std::string & data_path) override;

    // Train the model for a given number of epochs
    void train(const int &epochs, const int &batch_size, const double &learning_rate) override;

    // Evaluate the model's performance with the given test data and labels
    metrics::Metrics evaluate() override;

private:
    FFNet *ffnet;
    Dataset data;
    Metrics metrics;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

#endif // MODEL_FF_H
