#ifndef ORCHESTRATION_H
#define ORCHESTRATION_H

#include <vector>
#include <memory>
#include <client/client.h>
#include <server/server.h>
#include <metrics.hpp>
#include <spdlog/spdlog.h>

// Function prototypes for helper functions
std::vector<std::shared_ptr<Client>> initializeClients(const std::vector<std::string> &datasets_path);

class Orchestrator
{
public:
    Orchestrator(const std::string &datasets_path, const std::string &checkpoints_path);
    void run();

private:
    void logParams();
    void saveCheckpoint();
    std::vector<std::shared_ptr<Client>> sampleClients(int num_clients);
    metrics::Metrics evaluateClients(std::vector<std::shared_ptr<Client>> clients);

    size_t round_index = 0;
    std::vector<std::shared_ptr<Client>> clients;
    const std::string datasets_path;
    const std::string checkpoints_path;
    std::shared_ptr<Server> server;
};

#endif // ORCHESTRATION_H
