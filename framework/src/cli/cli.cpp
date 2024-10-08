#include "cli.hpp"

#include <iostream>
#include <vector>

#include <config/config.hpp>
#include <spdlog/spdlog.h>

config::ModelType get_model_type(std::vector<std::string> args);
std::string string_to_lower(const std::string &str);

void parse_args(const int argc, const char *argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if (argc < 2)
    {
        spdlog::info("No parameters provided. Running with default parameters.");
        return;
    }
    if (args[1] == "--help" || args[1] == "-h")
    {
        print_help(argv[0]);
        exit(EXIT_SUCCESS);
    }

    config::model_type = get_model_type(args);

    for (int i = 1; i < argc - 1; i++)
    {
        // Common model parameters
        if (args[i] == "--num-classes" || args[i] == "-nc")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid number of classes.");
                exit(EXIT_FAILURE);
            }
            config::parameters::num_classes = std::stoi(argv[i]);
            continue;
        }
        else if (args[i] == "--layer-units" || args[i] == "-lu")
        {
            std::vector<int> units = {};
            while (i + 1 < argc && args[i + 1][0] != '-')
            {
                units.push_back(std::stoi(argv[++i]));
            }
            if (units.size() < 2)
            {
                spdlog::error("Invalid number of units.");
                exit(EXIT_FAILURE);
            }
            config::parameters::units = units;
            std::string units_str = "";
            for (int unit : units)
            {
                units_str += std::to_string(unit) + " ";
            }
        }

        // Model FF specific parameters
        else if (args[i] == "--threshold" || args[i] == "-t")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid threshold.");
                exit(EXIT_FAILURE);
            }
            config::parameters::ff::threshold = std::stof(argv[i]);
        }
        else if (args[i] == "--loss-function" || args[i] == "-lf")
        {
            i++;
            std::string loss_function = string_to_lower(args[i]);
            if (loss_function == "ff")
            {
                config::parameters::ff::loss = LossType::LOSS_TYPE_FF;
            }
            else if (loss_function == "symba")
            {
                config::parameters::ff::loss = LossType::LOSS_TYPE_SYMBA;
            }
            else
            {
                spdlog::error("Invalid loss function.");
                exit(EXIT_FAILURE);
            }
        }
        else if (args[i] == "--beta1" || args[i] == "-b1")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid beta1.");
                exit(EXIT_FAILURE);
            }
            config::parameters::ff::beta1 = std::stof(argv[i]);
        }
        else if (args[i] == "--beta2" || args[i] == "-b2")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid beta2.");
                exit(EXIT_FAILURE);
            }
            config::parameters::ff::beta2 = std::stof(argv[i]);
        }

        // Training parameters
        else if (args[i] == "--learning-rate" || args[i] == "-lr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid learning rate.");
                exit(EXIT_FAILURE);
            }
            config::training::learning_rate = std::stof(argv[i]);
        }
        else if (args[i] == "--batch-size" || args[i] == "-bs")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid batch size.");
                exit(EXIT_FAILURE);
            }
            config::training::batch_size = std::stoi(argv[i]);
        }
        else if (args[i] == "--epochs" || args[i] == "-e")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid number of epochs.");
                exit(EXIT_FAILURE);
            }
            config::training::epochs = std::stoi(argv[i]);
        }

        // Orchestrator parameters
        else if (args[i] == "--num-clients" || args[i] == "-ncl")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid number of clients.");
                exit(EXIT_FAILURE);
            }
            config::orchestration::num_clients = std::stoi(argv[i]);
        }
        else if (args[i] == "--num-rounds" || args[i] == "-nr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid number of rounds.");
                exit(EXIT_FAILURE);
            }
            config::orchestration::num_rounds = std::stoi(argv[i]);
        }
        else if (args[i] == "--client-rate" || args[i] == "-cr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid c rate.");
                exit(EXIT_FAILURE);
            }
            config::orchestration::c_rate = std::stof(argv[i]);
        }
        else if (args[i] == "--checkpoint-rate" || args[i] == "-chr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid checkpoint rate.");
                exit(EXIT_FAILURE);
            }
            config::orchestration::checkpoint_rate = std::stof(argv[i]);
        }
        else if (args[i] == "--dataset" || args[i] == "-d")
        {
            i++;
            std::string dataset = string_to_lower(args[i]);
            if (dataset == "digits")
            {
                config::selected_dataset = config::dataset_digits;
            }
            else if (dataset == "mnist")
            {
                config::selected_dataset = config::dataset_mnist;
            }
            else if (dataset == "emnist")
            {
                config::selected_dataset = config::dataset_emnist;
            }
            else
            {
                spdlog::error("Invalid dataset.");
                exit(EXIT_FAILURE);
            }
        }
        else if (args[i] == "--log-level" || args[i] == "-ll")
        {
            i++;
            std::string log_level = string_to_lower(args[i]);
            if (log_level == "debug")
            {
                spdlog::set_level(spdlog::level::debug);
            }
            else if (log_level == "info")
            {
                spdlog::set_level(spdlog::level::info);
            }
            else if (log_level == "warn")
            {
                spdlog::set_level(spdlog::level::warn);
            }
            else if (log_level == "error")
            {
                spdlog::set_level(spdlog::level::err);
            }
            else
            {
                spdlog::error("Invalid log level.");
                exit(EXIT_FAILURE);
            }
        }
        else if (args[i] == "--threaded-mode" || args[i] == "-tm")
        {
            config::orchestration::threaded = true;
        }
    }
    if (args[argc - 1] == "--threaded-mode" || args[argc - 1] == "-tm")
    {
        config::orchestration::threaded = true;
    }
}

void print_help(std::string name)
{
    std::cout << "Usage:" << std::endl
              << name << " [OPTIONS]" << std::endl
              << "Options:" << std::endl
              << "--help, -h: Show this help message." << std::endl
              << "--model-type, -mt: Model type (bp, ff). Default: bp." << std::endl
              << "--num-classes, -nc: Number of classes in the dataset. Default: " << config::parameters::num_classes << "." << std::endl
              << "--layer-units, -lu: Number of units in each hidden layer. Default: [ ";
    for (int unit : config::parameters::units)
        std::cout << unit << " ";
    std::cout << "]" << std::endl
              << "--threshold, -t: Threshold for the FF model. Default: " << config::parameters::ff::threshold << "." << std::endl
              << "--loss-function, -lf: Loss function for the FF model (ff, symba). Default: ff." << std::endl
              << "--beta1, -b1: Beta1 for the FF model. Default: " << config::parameters::ff::beta1 << "." << std::endl
              << "--beta2, -b2: Beta2 for the FF model. Default: " << config::parameters::ff::beta2 << "." << std::endl
              << "--learning-rate, -lr: Learning rate for the training. Default: " << config::training::learning_rate << "." << std::endl
              << "--batch-size, -bs: Batch size for the training. Default: " << config::training::batch_size << "." << std::endl
              << "--epochs, -e: Number of epochs for the training. Default: " << config::training::epochs << "." << std::endl
              << "--num-clients, -ncl: Number of clients in the simulation. Default: " << config::orchestration::num_clients << "." << std::endl
              << "--num-rounds, -nr: Number of rounds in the simulation. Default: " << config::orchestration::num_rounds << "." << std::endl
              << "--client-rate, -cr: Client rate for the simulation. Default: " << config::orchestration::c_rate << "." << std::endl
              << "--checkpoint-rate, -chr: Checkpoint rate for the simulation. Default: " << config::orchestration::checkpoint_rate << "." << std::endl
              << "--dataset, -d: Dataset to use (digits, mnist, emnist). Default: << " << config::selected_dataset << "." << std::endl
              << "--log-level, -ll: Log level (debug, info, warn, error). Default: info." << std::endl
              << "--threaded-mode, -tm: Enable threaded mode for the orchestrator. Default: false." << std::endl;
}

config::ModelType get_model_type(std::vector<std::string> args)
{
    for (size_t i = 1; i < args.size() - 1; i++)
    {
        if (args[i] == "--model-type" || args[i] == "-mt")
        {
            std::string model = string_to_lower(args[i + 1]);
            if (model == "bp")
            {
                return config::ModelType::BP;
            }
            else if (model == "ff")
            {
                return config::ModelType::FF;
            }
            else
            {
                spdlog::error("Invalid model type.");
                exit(EXIT_FAILURE);
            }
        }
    }
    // if no model type is provided, default from config.hpp is used
    return config::model_type;
}

std::string string_to_lower(const std::string &str)
{
    std::string lower_str = str;
    for (char &c : lower_str)
    {
        c = std::tolower(c);
    }
    return lower_str;
}