#ifndef SERVER_H
#define SERVER_H

#include <client/client.hpp>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <spdlog/spdlog.h>
#include <set>

class Server
{
public:
    Server(const std::vector<std::shared_ptr<Client>>& clients, const std::string &global_dataset_path);
    Server(const std::vector<std::shared_ptr<Client>>& clients, const std::string &global_dataset_path, bool threaded);
    // Execute a federated learning round
    metrics::Metrics executeRound(int round_index, std::vector<std::shared_ptr<Client>> round_clients);

    // Updated clients since the last checkpoint
    std::set<std::shared_ptr<Client>> updated_clients;

    std::vector<metrics::Metrics> client_metrics;    

    // Server model
    std::shared_ptr<Model> model;
    
private:
    std::vector<std::shared_ptr<Client>> clients;
    std::vector<std::shared_ptr<Client>> round_clients;
    int max_clients;
    int round_index;
    bool threaded;

    void broadcast();
    void update_clients();
    std::vector<double> aggregate_models();
};

#endif // SERVER_H
