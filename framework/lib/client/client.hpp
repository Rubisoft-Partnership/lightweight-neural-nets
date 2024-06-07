#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <model/model.hpp>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <spdlog/spdlog.h>


class Client
{
public:
    int id;
    std::shared_ptr<Model> model;
    size_t dataset_size;
    const std::string data_path;
    std::vector<metrics::Metrics> history;
    std::vector<int> rounds; // list of rounds in which the client was updated

    Client(int id, std::shared_ptr<Model> model, const std::string &data_path);

    // Update the client with a new training round
    void update(int round_index, double learning_rate, size_t batch_size, size_t epochs);

    void logRounds() const;

    void logMetrics() const;

};

#endif // CLIENT_HPP
