#include "cli.hpp"

#include <iostream>
#include <vector>

#include <config/config.hpp>
#include <spdlog/spdlog.h>

config::ModelType get_model_type(std::vector<std::string> args);
std::string string_to_lower(const std::string &str);

void parse_args(const int argc, const char *argv[])
{
    spdlog::debug("Parsing command line arguments.");
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
            config::num_classes = std::stoi(argv[i]);
            continue;
        }
        if (args[i] == "--layer-units" || args[i] == "-lu")
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
            config::model_bp_parameters.units = units;
            config::model_ff_parameters.units = units;
            std::string units_str = "";
            for (int unit : units)
            {
                units_str += std::to_string(unit) + " ";
            }
            spdlog::debug("Layer units: {}", units_str);
        }

        // Model FF specific parameters
        if (args[i] == "--threshold" || args[i] == "-t")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid threshold.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Threshold: {}", argv[i]);
            config::model_ff_parameters.threshold = std::stof(argv[i]);
        }
        if (args[i] == "--loss-function" || args[i] == "-lf")
        {
            i++;
            std::string loss_function = string_to_lower(args[i]);
            if (loss_function == "ff")
            {
                config::model_ff_parameters.loss = LossType::LOSS_TYPE_FF;
            }
            else if (loss_function == "symba")
            {
                config::model_ff_parameters.loss = LossType::LOSS_TYPE_SYMBA;
            }
            else
            {
                spdlog::error("Invalid loss function.");
                exit(EXIT_FAILURE);
            }
        }
        if (args[i] == "--beta1" || args[i] == "-b1")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid beta1.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Beta1: {}", argv[i]);
            config::model_ff_parameters.beta1 = std::stof(argv[i]);
        }
        if (args[i] == "--beta2" || args[i] == "-b2")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid beta2.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Beta2: {}", argv[i]);
            config::model_ff_parameters.beta2 = std::stof(argv[i]);
        }

        // Training parameters
        if (args[i] == "--learning-rate" || args[i] == "-lr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid learning rate.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Learning rate: {}", argv[i]);
            config::training_parameters.learning_rate = std::stof(argv[i]);
        }
        if (args[i] == "--batch-size" || args[i] == "-bs")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid batch size.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Batch size: {}", argv[i]);
            config::training_parameters.batch_size = std::stoi(argv[i]);
        }
        if (args[i] == "--epochs" || args[i] == "-e")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid number of epochs.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Number of epochs: {}", argv[i]);
            config::training_parameters.epochs = std::stoi(argv[i]);
        }

        // Orchestrator parameters
        if (args[i] == "--num-clients" || args[i] == "-nc")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid number of clients.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Number of clients: {}", argv[i]);
            config::orchestrator::num_clients = std::stoi(argv[i]);
        }
        if (args[i] == "--num-rounds" || args[i] == "-nr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid number of rounds.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Number of rounds: {}", argv[i]);
            config::orchestrator::num_rounds = std::stoi(argv[i]);
        }
        if (args[i] == "--client-rate" || args[i] == "-cr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid c rate.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("C rate: {}", argv[i]);
            config::orchestrator::c_rate = std::stof(argv[i]);
        }
        if (args[i] == "--checkpoint-rate" || args[i] == "-chr")
        {
            i++;
            if (args[i][0] == '-')
            {
                spdlog::error("Invalid checkpoint rate.");
                exit(EXIT_FAILURE);
            }
            spdlog::debug("Checkpoint rate: {}", argv[i]);
            config::orchestrator::checkpoint_rate = std::stof(argv[i]);
        }
    }
}

void print_help(std::string name)
{
    std::cout << "Usage:" << std::endl
              << name << " [OPTIONS]" << std::endl;
}

config::ModelType get_model_type(std::vector<std::string> args)
{
    for (int i = 1; i < args.size() - 1; i++)
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
                std::cout << "Invalid model type." << std::endl;
                print_help(args[0]);
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