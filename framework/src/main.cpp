#include <iostream>
#include <string>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <orchestration/orchestration.hpp>
#include <config/config.hpp>

void init_logger()
{
    try
    {
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
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    (void)argc; // Unused parameter
    (void)argv; // Unused parameter

    // Initialize configurations.
    config::init_config();

    // Set up logger
    init_logger();

    // Create the simulation folder
    if (!std::filesystem::create_directories(config::simulation_path))
    {
        spdlog::error("Failed to create simulation directory at: {}", config::simulation_path);
        return EXIT_FAILURE;
    }
    // Create the checkpoints folder
    if (!std::filesystem::create_directories(config::checkpoints_path))
    {
        spdlog::error("Failed to create checkpoints directory at: {}", config::checkpoints_path);
        return EXIT_FAILURE;
    }

    try
    {
        spdlog::info("Starting Federated Learning Orchestrator...");
        // Initialize the orchestrator
        Orchestrator orchestrator(config::datasets_path + config::dataset_digits, config::checkpoints_path);

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
