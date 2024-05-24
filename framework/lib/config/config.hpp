#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace config
{
    void init_config();

    const std::string datasets_folder = "/dataset/federated/";        // relative path for datasets
    const std::string dataset_digits = "/digits/";                    // digits dataset folder name
    const std::string dataset_mnist = "/mnist/";                      // mnist dataset folder name
    const std::string simulations_folder = "/framework/simulations/"; // relative simulations path
    const std::string checkpoints_folder = "/checkpoints/";           // checkpoints folder name
    const std::string logs_folder = "/framework/logs/";               // relative logs path
    const std::string logger_name = "framework_logger";               // logger name

    extern std::string basepath;             // project base path
    extern std::string datasets_path;        // absolute path to the datasets
    extern std::string simulation_path;      // absolute path to the current simulation
    extern std::string checkpoints_path;     // absolute path to the current simulation checkpoints
    extern std::string log_path;             // absolute path to the current simulation log file
    extern std::string simulation_timestamp; // current simulation timestamp
}

#endif // CONFIG_H