#ifndef MODEL_BP_H
#define MODEL_BP_H

#include "../framework/lib/model/model.hpp"
#include <vector>
#include <random>
#include <metrics.hpp>
#include <tiny_dnn/tiny_dnn.h>

struct ModelBPParameters
{
    std::vector<int> units;
};
    

class ModelBP : public Model
{
public:
    // Destructor
    virtual ~ModelBP() {}

    // Initialize the model with necessary parameters or configurations
    void build(const std::string & data_path) override;

    // Train the model for a given number of epochs
    void train(const int &epochs, const int &batch_size, const double &learning_rate) override;

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

    // Set model's parameters
    void set_parametersBP(const ModelBPParameters &parameters);

private:
    tiny_dnn::network<tiny_dnn::sequential> bpnet;
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;
    tiny_dnn::adam optimizer;

    const tiny_dnn::float_t min_scale = -1.0;
    const tiny_dnn::float_t max_scale = 1.0;
    const int x_padding = 0;
    const int y_padding = 0;
};

#endif // MODEL_BP_H
