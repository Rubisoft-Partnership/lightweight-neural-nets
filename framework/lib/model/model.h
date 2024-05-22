#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <metrics.hpp>

class Model
{
public:
    // Virtual destructor to ensure proper cleanup of derived classes
    virtual ~Model() {}

    // Initialize the model with necessary parameters or configurations
    virtual void build(const std::vector<int> &units, const std::string &data_path) = 0;

    // Train the model for a given number of epochs
    virtual void train(const int &epochs, const int &batch_size, const double &learning_rate) = 0;

    // Evaluate the model's performance with the given test data and labels
    virtual metrics::Metrics evaluate() = 0;

protected:
    std::vector<int> units;
};

#endif // MODEL_H
