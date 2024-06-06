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