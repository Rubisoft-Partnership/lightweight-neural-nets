#include <fstream>
#include <string>
#include <algorithm>

#include <client/client.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <metrics-logger/metrics-logger.hpp>


Client::Client(int id, std::shared_ptr<Model> model, const std::string &data_path)
    : id(id), model(model), data_path(data_path)
{
    // Initialize model with given units and data path
    model->build(data_path);

    // Calculate and store the dataset size
    dataset_size = model->dataset_size;

    // Log the initialization
    spdlog::info("Initialized client {}.", id);
    spdlog::debug("Model data path: {}.", data_path);
    spdlog::debug("Model dataset size: {} samples.", dataset_size);
}

void Client::update(int round_index, double learning_rate, size_t batch_size, size_t epochs)
{
    spdlog::info("Updating client: {}.", id);
    spdlog::debug("Round index: {}.", round_index);

    if (dataset_size == 0)
    {
        spdlog::error("Empty dataset for client {}.", id);
        exit(EXIT_FAILURE);
    }

    if (dataset_size == 0)
    {
        spdlog::error("Empty dataset for client {}.", id);
        exit(EXIT_FAILURE);
    }

    int epoch = 0;
    auto on_enumerate_epoch = [&]()
    {
        metrics::Metrics metrics = model->evaluate();
        log_metrics(round_index, id, epoch, DatasetType::LOCAL, metrics);
        spdlog::debug("Client {} epoch {} accuracy: {}, loss {}.", id, epoch, metrics.accuracy, metrics.loss);
        epoch++;
    };
    // Train the model
    model->train(epochs, batch_size, learning_rate, on_enumerate_epoch);

    // Evaluate the model and store the metrics
    auto metrics = model->evaluate();
    history.push_back(metrics);

    // Update round count and store the round index
    rounds.push_back(round_index);

    spdlog::info("Done updating client: {}.", id);
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
