#include <fstream>
#include <string>
#include <algorithm>

#include <client/client.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <model-bp.hpp>
#include <model-ff.hpp>

namespace config
{
    ModelBPParameters model_bp_parameters;
    ModelFFParameters model_ff_parameters;

    extern ModelType model_type;
}

Client::Client(int id, std::shared_ptr<Model> model, const std::string &data_path)
    : id(id), model(model), data_path(data_path)
{
    ModelBP *bp = NULL;
    ModelFF *ff = NULL;
    switch (config::model_type)
    {
    case config::ModelType::BP:
        bp = dynamic_cast<ModelBP *>(model.get());
        bp->set_parametersBP(config::model_bp_parameters);
        break;
    case config::ModelType::FF:
        ff = dynamic_cast<ModelFF *>(model.get());
        ff->set_parametersFF(config::model_ff_parameters);
        break;
    }
    // Initialize model with given units and data path
    model->build(data_path);

    // Calculate and store the dataset size
    dataset_size = model->dataset_size;
    if (dataset_size == -1)
    {
        spdlog::error("Invalid dataset size for client {}.", id);
        exit(EXIT_FAILURE);
    }

    // Log the initialization
    spdlog::info("Initialized client {}.", id);
    spdlog::debug("Model data path: {}.", data_path);
    spdlog::debug("Model dataset size: {} samples.", dataset_size);
}

void Client::update(int round_index, double learning_rate, size_t batch_size, size_t epochs)
{
    spdlog::info("Updating client: {}.", id);
    spdlog::debug("Round index: {}.", round_index);
    spdlog::debug("Learning rate: {}.", learning_rate);
    spdlog::debug("Batch size: {}.", batch_size);

    // Train the model
    model->train(epochs, batch_size, learning_rate);

    // Evaluate the model and store the metrics
    auto metrics = model->evaluate();
    history.push_back(metrics);

    // Update round count and store the round index
    rounds.push_back(round_index);

    spdlog::info("Done updating client: {}.", id);
    spdlog::debug("Client {} accuracy: {}.", id, metrics.accuracy);
}

void Client::logRounds() const
{
    std::ostringstream oss;
    for (size_t i = 0; i < rounds.size(); ++i)
    {
        if (i != 0)
            oss << ", ";
        oss << rounds[i];
    }
    spdlog::info("Client {} was updated in rounds: {}.", id, oss.str());
}

void Client::logMetrics() const
{
    for (size_t i = 0; i < history.size(); ++i)
    {
        spdlog::info("Client {} metrics for round {}: {}.", id, rounds[i], history[i].toString());
    }
}
