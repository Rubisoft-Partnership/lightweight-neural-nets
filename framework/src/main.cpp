#include <iostream>
#include <string>
#include <spdlog/spdlog.h>
#include <orchestration/orchestration.hpp>

const std::string datasets_path = "datasets";
const std::string checkpoints_path = "checkpoints";

int main(int argc, char *argv[])
{
    (void)argc; // Unused parameter
    (void)argv; // Unused parameter

    // Set up logger
    spdlog::set_level(spdlog::level::debug); // Set default logging level to info
    spdlog::info("Starting Federated Learning Orchestrator...");

    try
    {
        // Initialize the orchestrator
        Orchestrator orchestrator(datasets_path, checkpoints_path);

        // Run the orchestrator
        orchestrator.run();

        spdlog::info("Federated Learning Orchestrator finished successfully.");
    }
    catch (const std::exception& e)
    {
        spdlog::error("An error occurred: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
