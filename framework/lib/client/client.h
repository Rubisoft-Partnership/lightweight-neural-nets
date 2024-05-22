#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <model/model.h>
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
    int dataset_size;
    const std::string data_path;
    std::vector<metrics::Metrics> history;
    std::vector<int> rounds; // list of rounds in which the client was updated

    Client(int id, const std::string &data_path);

    // Update the client with a new training round
    void update(int round_index, double learning_rate, size_t batch_size, size_t epochs);

    void logRounds() const;

    void logMetrics() const;

private:
    // Calculate the size of the dataset
    int calculateDatasetSize();
};

#endif // CLIENT_HPP
