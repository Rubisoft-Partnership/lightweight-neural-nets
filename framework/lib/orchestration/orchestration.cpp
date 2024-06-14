#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <regex>

#include <orchestration/orchestration.hpp>
#include <spdlog/spdlog.h>
#include <model-ff.hpp>
#include <model-bp.hpp>
#include <config/config.hpp>

namespace fs = std::filesystem;

using namespace config::orchestrator;

static std::vector<std::string> listFolders(const std::string &folder, const std::string &match);

Orchestrator::Orchestrator(const std::string &datasets_path, const std::string &checkpoints_path) : datasets_path(datasets_path),
                                                                                                    checkpoints_path(checkpoints_path)
{
    // Search datasets folders
    std::vector<std::string> data = listFolders(datasets_path, "^client-\\d+$");
    if (data.empty())
    {
        spdlog::error("No datasets found in folder: {}.", datasets_path);
        exit(EXIT_FAILURE);
    }

    // Initialize clients
    clients = initializeClients(data);

    // Initialize server
    server = std::make_shared<Server>(clients, datasets_path + config::global_dataset);
}

std::vector<std::shared_ptr<Client>> Orchestrator::sampleClients()
{
    spdlog::info("Sampling {} clients...", num_clients);
    // Randomly select a subset of clients
    std::vector<std::shared_ptr<Client>> selected_clients;
    std::sample(clients.begin(), clients.end(),
                std::back_inserter(selected_clients),
                std::max(static_cast<size_t>(1), static_cast<size_t>(c_rate * num_clients)),
                std::mt19937{std::random_device{}()});

    // Log the id of all selected clients
    for (const auto &client : selected_clients)
        spdlog::info("Selected client with id: {}.", client->id);
    return selected_clients;
}

void Orchestrator::run()
{
    for (round_index = 0; round_index < num_rounds; ++round_index)
    {
        spdlog::info("Running communication round: {}.", round_index);
        std::vector<std::shared_ptr<Client>> round_clients = sampleClients();

        metrics::Metrics new_model_metrics = server->executeRound(round_index, round_clients);
        spdlog::info("Updated model metrics:\n{}", new_model_metrics.toString());

        spdlog::info("Starting round clients evaluation.");
        metrics::Metrics round_avg_metrics = evaluateClients(round_clients);
        spdlog::info("Round average metrics:\n{}", round_avg_metrics.toString());

        spdlog::info("Starting global evaluation.");
        metrics::Metrics global_avg_metrics = evaluateClients(clients);
        spdlog::info("Global average metrics:\n{}", global_avg_metrics.toString());

        if (round_index % std::max(static_cast<int>(num_rounds * checkpoint_rate), 1) == 0 && round_index > 0)
            saveCheckpoint();
    }

    for (auto &client : clients)
        client->logRounds();
}

void Orchestrator::saveCheckpoint()
{
    const std::string &path = checkpoints_path;
    spdlog::info("Saving checkpoint at round: {}.", round_index);

    // Create the checkpoints folder if it does not exist
    if (!fs::exists(path))
    {
        spdlog::info("Creating checkpoints folder: {}.", path);
        fs::create_directory(path);
    }
    else if (!fs::is_directory(path))
    {
        spdlog::error("Checkpoints path is not a directory: {}.", path);
        exit(EXIT_FAILURE);
    }

    // Create the checkpoint folder for the current round
    std::string round_folder = path + "/checkpoint-round-" + std::to_string(round_index);
    if (fs::exists(round_folder))
    {
        spdlog::warn("Saving checkpoint to an already existing folder: {}.", round_folder);
    }
    else
    {
        spdlog::info("Creating checkpoint folder: {}.", round_folder);
        fs::create_directory(round_folder);
    }

    for (auto &client : clients)
    {
        client->model->save(round_folder + "/model-client-" + std::to_string(client->id) + ".bin");
    }
}

std::vector<std::shared_ptr<Client>> initializeClients(const std::vector<std::string> &datasets_path)
{
    spdlog::info("Initializing clients...");
    spdlog::debug("Number of clients: {}.", num_clients);
    spdlog::debug("Number of datasets: {}.", datasets_path.size());
    if (datasets_path.size() < num_clients)
        spdlog::warn("Number of datasets is less than the number of clients. Some clients will share the same dataset. This should be avoided.");

    std::vector<std::shared_ptr<Client>> clients;
    for (size_t i = 0; i < num_clients; ++i)
    {
        std::shared_ptr<ModelFF> modelff;
        std::shared_ptr<ModelBP> modelbp;
        std::shared_ptr<Client> client;
        switch (config::model_type)
        {
            case config::ModelType::FF:
                modelff = std::make_shared<ModelFF>();
                client = std::make_shared<Client>(i, modelff, datasets_path[i % datasets_path.size()]);
                break;
            case config::ModelType::BP:
                modelbp = std::make_shared<ModelBP>();
                client = std::make_shared<Client>(i, modelbp, datasets_path[i % datasets_path.size()]);
                break;
            default:
                spdlog::error("Invalid model type.");
                exit(EXIT_FAILURE);
        }
        clients.push_back(client);
    }
    return clients;
}

metrics::Metrics Orchestrator::evaluateClients(std::vector<std::shared_ptr<Client>> clients)
{
    // Write a log info message with all the id of the clients
    spdlog::info("Evaluating {} clients: {}.",
                 clients.size(),
                 [&]()
                 {
                     std::string ids;
                     for (const auto &client : clients)
                     {
                         ids += std::to_string(client->id) + " ";
                     }
                     return ids;
                 }());

    std::vector<metrics::Metrics> round_metrics;
    for (const auto &client : clients)
    {
        round_metrics.push_back(client->model->evaluate());
        spdlog::debug("Client {} accuracy: {}.", client->id, round_metrics.back().accuracy);
    }

    return metrics::mean(round_metrics);
}

static std::vector<std::string> listFolders(const std::string &folder, const std::string &match)
{
    spdlog::info("Listing folders in folder: {}.", folder);
    std::vector<std::string> folders;
    std::regex pattern(match);

    try
    {
        for (const auto &entry : fs::directory_iterator(folder))
        {
            if (entry.is_directory())
            {
                std::string foldername = entry.path().filename().string();
                if (std::regex_match(foldername, pattern))
                {
                    folders.push_back(folder + foldername);
                }
            }
        }
    }
    catch (const fs::filesystem_error &err)
    {
        spdlog::error("Filesystem error: {}.", err.what());
    }
    catch (const std::regex_error &err)
    {
        spdlog::error("Regex error: {}.", err.what());
    }
    return folders;
}

std::string findNextFolder(const std::string &parent_folder)
{
    spdlog::info("Finding next folder in parent folder: {}.", parent_folder);

    // Check if the parent folder ends with a slash
    std::string corrected_parent_folder = parent_folder;
    if (!corrected_parent_folder.empty() && corrected_parent_folder.back() != '/')
        corrected_parent_folder += '/';

    std::string next_folder;
    int i = 0;
    while (true)
    {
        next_folder = corrected_parent_folder + std::to_string(i);
        if (!fs::exists(next_folder))
        {
            break;
        }
        i++;
    }
    return next_folder;
}
