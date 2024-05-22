#ifndef SERVER_H
#define SERVER_H

#include <client/client.h>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <spdlog/spdlog.h>

class Server
{
public:
    Server(const std::vector<std::shared_ptr<Client>>& clients);

    // Execute a federated learning round
    metrics::Metrics executeRound(int round_index, std::vector<std::shared_ptr<Client>> round_clients);

private:
    std::vector<std::shared_ptr<Client>> clients;
    int max_clients;
    bool threaded;
};

#endif // SERVER_H
