#include <iostream>
#include <string>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <orchestration/orchestration.hpp>
#include <config/config.hpp>
#include <metrics-logger/metrics-logger.hpp>

#include "cli/cli.hpp"

void init_logger()
{
    try
    {
        if (std::filesystem::exists(config::log_path))
        {
            throw std::runtime_error("Log file already exists.");
        }
        // Create a file sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(config::log_path, true);
        // Create a console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        // Combine the sinks into a multi-sink logger
        std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
        auto logger = std::make_shared<spdlog::logger>(config::logger_name, sinks.begin(), sinks.end());
        // Set the logger's level to debug (default is info)
        logger->set_level(spdlog::level::debug);
        // Register the logger to make it globally accessible and set it as default
        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);
        // Set the logger's pattern
        logger->set_pattern("[%X] [%^%L%$] [%@:%#%&%!] %v");
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(const int argc, const char *argv[])
{
    // Initialize configurations.
    config::init_config();

    // Initialize the logger. Retry up to 5 times if a concurrency issue occurs.
#define MAX_ATTEMPTS 5
    // Create the simulation folder
    int numAttempts = 0;
    while (numAttempts < MAX_ATTEMPTS)
    {
        try
        {
            init_logger();
            break;
        }
        catch (const std::exception &e)
        {
            std::cerr << "An error occurred while initializing the logger: " << e.what() << std::endl
                      << "Retrying..." << std::endl;
            config::init_config();
            numAttempts++;
        }
    }

    // Parse command line arguments
    parse_args(argc, argv);

    if (!std::filesystem::create_directories(config::simulation_path))
    {
        spdlog::error("Failed to create simulation directory at: {}", config::simulation_path);
        return EXIT_FAILURE;
    }
    spdlog::info("Simulation directory created at: {}", config::simulation_path);

    // Create the checkpoints folder
    if (!std::filesystem::create_directories(config::checkpoints_path))
    {
        spdlog::error("Failed to create checkpoints directory at: {}", config::checkpoints_path);
        return EXIT_FAILURE;
    }

    config::log_simulation_params();
    config::save_config_to_file();
    init_metrics_logger();

    try
    {
        spdlog::info("Starting Federated Learning Orchestrator...");
        // Initialize the orchestrator
        Orchestrator orchestrator(config::datasets_path + config::selected_dataset, config::checkpoints_path, config::orchestration::threaded);

        // Run the orchestrator
        orchestrator.run();

        spdlog::info("Federated Learning Orchestrator finished successfully.");
    }
    catch (const std::exception &e)
    {
        spdlog::error("An error occurred: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
