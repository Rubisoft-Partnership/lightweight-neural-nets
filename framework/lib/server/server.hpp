#ifndef SERVER_H
#define SERVER_H

#include <client/client.hpp>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <spdlog/spdlog.h>

class Server
{
public:
    Server(const std::vector<std::shared_ptr<Client>>& clients, const std::string &global_dataset_path);
    // Execute a federated learning round
    metrics::Metrics executeRound(int round_index, std::vector<std::shared_ptr<Client>> round_clients);

private:
    std::vector<std::shared_ptr<Client>> clients;
    std::vector<std::shared_ptr<Client>> round_clients;
    int max_clients;
    int round_index;
    std::shared_ptr<Model> model;
    bool threaded;

    void broadcast();
    void update_clients();
    std::vector<double> aggregate_models();
};

#endif // SERVER_H
