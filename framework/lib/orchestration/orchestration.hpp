#ifndef ORCHESTRATION_H
#define ORCHESTRATION_H

#include <vector>
#include <memory>
#include <client/client.hpp>
#include <server/server.hpp>
#include <metrics.hpp>
#include <spdlog/spdlog.h>

// Function prototypes for helper functions
std::vector<std::shared_ptr<Client>> initializeClients(const std::vector<std::string> &datasets_path);

class Orchestrator
{
public:
    Orchestrator(const std::string &datasets_path, const std::string &checkpoints_path, bool threaded = false);
    void run();

private:
    void saveCheckpoint();
    std::vector<std::shared_ptr<Client>> sampleClients();
    metrics::Metrics evaluateClients(std::vector<std::shared_ptr<Client>> clients);

    size_t round_index = 0;
    std::vector<std::shared_ptr<Client>> clients;
    const std::string datasets_path;
    const std::string checkpoints_path;
    std::shared_ptr<Server> server;
    bool threaded;
};

#endif // ORCHESTRATION_H
