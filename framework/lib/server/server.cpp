#include <server/server.hpp>
#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <config/config.hpp>

namespace config
{
    extern TrainingParameters training_parameters;
}


// TODO: implement threaded mode
Server::Server(const std::vector<std::shared_ptr<Client>> &clients)
    : clients(clients), max_clients(clients.size()), threaded(false)
{
    spdlog::info("Initialized server with threaded mode: {}.", threaded ? "enabled" : "disabled");
}

metrics::Metrics Server::executeRound(int round_index, std::vector<std::shared_ptr<Client>> round_clients)
{
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

    // Update clients
    for (auto &client : round_clients)
    {
        client->update(round_index, config::training_parameters.learning_rate, config::training_parameters.batch_size, config::training_parameters.epochs);
    }
    spdlog::info("Done updating clients.");

    spdlog::info("Aggregating updated clients' models.");
    std::vector<std::vector<double>> models;
    std::vector<int> dataset_sizes;
    for (auto &client : round_clients)
    {
        models.push_back(client->model->get_weights());
        dataset_sizes.push_back(client->dataset_size);
    }

    // Compute weighted average of models
    std::vector<double> new_model(models[0].size(), 0.0);
    int total_size = std::accumulate(dataset_sizes.begin(), dataset_sizes.end(), 0);
    for (size_t i = 0; i < models.size(); ++i)
    {
        for (size_t j = 0; j < models[i].size(); ++j)
        {
            new_model[j] += models[i][j] * (static_cast<double>(dataset_sizes[i]) / total_size);
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
            diff += std::pow(models[i][j] - new_model[j], 2);
        }
        model_diffs.push_back(std::sqrt(diff));
    }
    // Compute mean weight standard deviation
    double mean_weight_std = std::accumulate(model_diffs.begin(), model_diffs.end(), 0.0) / model_diffs.size();
    spdlog::info("Mean weight standard deviation: {}.", mean_weight_std);


    // Set the new model weights to all clients
    for (auto &client : round_clients)
    {
        client->model->set_weights(new_model);
    }
    spdlog::info("Updated model broadcast complete.");

    // Test new model
    metrics::Metrics new_model_metrics = clients[0]->model->evaluate();
    return new_model_metrics;
}

