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
    metrics::Metrics evaluate();

    // Get the model's weights
    std::vector<double> get_weights() const;

    // Set the model's weights
    void set_weights(const std::vector<double> &weights);

    // Save the model's weights to a file
    void save(const std::string filename);

    // Load the model's weights from a file
    void load(const std::string filename);

private:
    FFNet *ffnet;
    Dataset data;

    int max_units;
    int input_size;
};

#endif // MODEL_FF_H
