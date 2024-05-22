#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <model/model.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <spdlog/spdlog.h>

// Logger placeholder
#define logger_info(msg) std::cout << msg << std::endl

class Client
{
public:
    Client(int id, const std::vector<int> &units, const std::string &data_path);

    // Update the client with a new training round
    void update(int round_index, double learning_rate, size_t batch_size, size_t epochs);

    void Client::logRounds() const;
    
    void Client::logMetrics() const;

private:
    int id;
    const std::string data_path;
    std::shared_ptr<Model> model;
    int dataset_size;
    std::vector<metrics::Metrics> history;
    std::vector<int> rounds; // list of rounds in which the client was updated

    // Calculate the size of the dataset
    int calculateDatasetSize();
};

#endif // CLIENT_HPP
