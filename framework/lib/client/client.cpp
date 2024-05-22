#include <fstream>
#include <string>
#include <algorithm>

#include <client/client.h>
#include <spdlog/spdlog.h>

// TODO: move this to a configuration file.
const std::vector<int> &units = {784, 100, 100, 100};

Client::Client(int id, std::shared_ptr<Model> model, const std::string &data_path)
    : id(id), data_path(data_path)
{
    // Initialize model with given units and data path
    model->build(units, data_path);

    // Calculate and store the dataset size
    dataset_size = calculateDatasetSize();
    if (dataset_size == -1)
    {
        spdlog::error("Failed to calculate dataset size for client {}.", id);
        exit(EXIT_FAILURE);
    }

    // Log the initialization
    spdlog::info("Initialized client {}.", id);
    // Format units as a string
    std::ostringstream oss;
    for (size_t i = 0; i < units.size(); ++i)
    {
        if (i != 0)
            oss << ", ";
        oss << units[i];
    }
    spdlog::debug("Model units: {}.", oss.str());
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
    spdlog::debug("Metrics: {}.", metrics.toString());
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

int Client::calculateDatasetSize()
{
    std::ifstream file(data_path);
    if (!file.is_open())
    {
        spdlog::error("Failed to open data file: {}.", data_path);
        return -1;
    }

    int lineCount = std::count(std::istreambuf_iterator<char>(file),
                               std::istreambuf_iterator<char>(), '\n');

    spdlog::debug("Calculated dataset size: {}.", lineCount);

    return lineCount;
}