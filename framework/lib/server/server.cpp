#include <server/server.hpp>
#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <config/config.hpp>
#include <model-ff.hpp>
#include <model-bp.hpp>
#include <metrics-logger/metrics-logger.hpp>
#include <thread>
#include "server.hpp"

using namespace config::training;

Server::Server(const std::vector<std::shared_ptr<Client>> &clients, const std::string &global_dataset_path, bool threaded)
    : clients(clients), max_clients(clients.size()), threaded(threaded)
{
    client_metrics = std::vector<metrics::Metrics>(max_clients);
    // Initialize server model weights with the first client model weights
    if (config::model_type == config::ModelType::FF)
    {
        model = std::make_shared<ModelFF>();
    }
    else if (config::model_type == config::ModelType::BP)
    {
        model = std::make_shared<ModelBP>();
    }
    else
    {
        spdlog::error("Model type not supported.");
        exit(EXIT_FAILURE);
    }
    model->build(global_dataset_path);
    spdlog::info("Initialized server with threaded mode: {}.", threaded ? "enabled" : "disabled");
}

Server::Server(const std::vector<std::shared_ptr<Client>> &clients, const std::string &global_dataset_path)
    : Server(clients, global_dataset_path, false) {}

metrics::Metrics Server::executeRound(int round_index, std::vector<std::shared_ptr<Client>> round_clients)
{
    this->round_clients = round_clients;
    this->round_index = round_index;

    spdlog::info("Updating selected {} clients: {}.",
                 round_clients.size(),
                 [&]()
                 {
                     std::string ids;
                     for (const auto &client : round_clients)
                     {
                         ids += std::to_string(client->id) + " ";
                     }
                     return ids;
                 }());

    // Set the new model weights to all clients
    broadcast();

    // Update clients
    update_clients();

    // Add round clients to the updated clients set
    updated_clients.insert(round_clients.begin(), round_clients.end());

    // Aggregate models
    model->set_weights(aggregate_models());

    spdlog::info("Server model updated with the aggregated model.");

    // Test new model
    metrics::Metrics new_model_metrics = model->evaluate();
    log_metrics(round_index, -1, -1, DatasetType::GLOBAL, new_model_metrics);
    return new_model_metrics;
}

void Server::broadcast()
{
    std::vector<std::thread> threads;
    // Reserve space only if threaded mode is enabled
    if (threaded)
        threads.reserve(round_clients.size());

    // Use a shared pointer for model_weights
    auto model_weights = std::make_shared<std::vector<double>>(model->get_weights());

    // Function to set weights for a client
    auto set_weights_for_client = [&](Client& client) {
        client.model->set_weights(*model_weights);
    };
   
    for (auto& client : round_clients) {
        if (threaded) {
            // Create a thread for each client
            threads.emplace_back(std::thread(set_weights_for_client, std::ref(*client)));
        } else {
            // Set weights directly if not using threads
            set_weights_for_client(*client);
        }
    }

    // Join all threads if threaded
    if (threaded) {
        for (auto& thread : threads) {
            thread.join();
        }
    }

    spdlog::info("Server model broadcast completed.");
}

void Server::update_clients()
{
    std::vector<std::thread> threads;
    // Reserve space only if threaded mode is enabled
    if (threaded)
        threads.reserve(round_clients.size());

    for (auto &client : round_clients)
    {
        if (threaded)
            threads.emplace_back([&client, this]()
                                 { 
                                    client->update(round_index, learning_rate, batch_size, epochs); });
        else
            client->update(round_index, learning_rate, batch_size, epochs);
    }

    if (threaded)
        for (auto &thread : threads)
            thread.join();

    spdlog::info("Done updating clients.");
}

std::vector<double> Server::aggregate_models()
{
    spdlog::info("Aggregating updated clients models.");
    std::vector<std::vector<double>> models;
    std::vector<int> dataset_sizes;
    for (auto &client : round_clients)
    {
        models.push_back(client->model->get_weights());
        dataset_sizes.push_back(client->dataset_size);
    }

    // Compute weighted average of models
    std::vector<double> new_model_weights(models[0].size(), 0.0);
    int total_size = std::accumulate(dataset_sizes.begin(), dataset_sizes.end(), 0);
    for (size_t i = 0; i < models.size(); ++i)
    {
        for (size_t j = 0; j < models[i].size(); ++j)
        {
            new_model_weights[j] += models[i][j] * (static_cast<double>(dataset_sizes[i]) / total_size);
        }
    }

    // TODO: consider removing this computation to speed up round execution
    // Compute model standard deviation
    std::vector<double> model_diffs;
    for (size_t i = 0; i < models.size(); ++i)
    {
        double diff = 0.0;
        for (size_t j = 0; j < models[i].size(); ++j)
        {
            diff += std::pow(models[i][j] - new_model_weights[j], 2);
        }
        model_diffs.push_back(std::sqrt(diff));
    }
    // Compute mean weight standard deviation
    double mean_weight_std = std::accumulate(model_diffs.begin(), model_diffs.end(), 0.0) / model_diffs.size();
    spdlog::info("Mean weight standard deviation: {}.", mean_weight_std);

    return new_model_weights;
}
