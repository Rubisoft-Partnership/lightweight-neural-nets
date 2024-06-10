#ifndef MODEL_FF_H
#define MODEL_FF_H

#include "../../framework/lib/model/model.hpp"
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
    void build(const std::string & data_path) override;

    // Train the model for a given number of epochs
    void train(const int &epochs, const int &batch_size, const double &learning_rate, std::function<void()> on_enumerate_epoch) override;

    // Evaluate the model's performance with the given test data and labels
    metrics::Metrics evaluate() override;

    // Get the model's weights
    std::vector<double> get_weights() const override;

    // Set the model's weights
    void set_weights(const std::vector<double> &weights) override;

    // Save the model's weights to a file
    void save(const std::string filename) override;

    // Load the model's weights from a file
    void load(const std::string filename) override;

private:
    FFNet *ffnet;
    Dataset data;

    float threshold;
    float beta1, beta2;
    LossType loss;
};

#endif // MODEL_FF_H
