#include "metrics-logger.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <config/config.hpp>

#define METRICS_LOGGER_NAME "metrics_logger"

void init_metrics_logger()
{
    // Create a file sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(config::simulation_path + "/metrics.csv", true);
    file_sink->set_pattern("%v");
    // Combine the sinks into a multi-sink logger
    std::vector<spdlog::sink_ptr> sinks{file_sink};
    auto logger = std::make_shared<spdlog::logger>(METRICS_LOGGER_NAME, sinks.begin(), sinks.end());
    // Register the logger to make it globally accessible
    spdlog::register_logger(logger);
    spdlog::debug("Metrics logger initialized");
    logger->info("round_num,client_id,epoch,dataset_type,accuracy,average_f1_score,average_precision,average_recall,loss");
}

void log_metrics(const int round_num, const int client_id, const int epoch, const DatasetType dataset_type, const metrics::Metrics &metrics)
{
    // Get the metrics logger
    auto logger = spdlog::get(METRICS_LOGGER_NAME);
    // Log the metrics
    logger->info("{},{},{},{},{},{},{},{},{}", round_num, client_id, epoch, static_cast<int>(dataset_type), metrics.accuracy, metrics.average_f1_score, metrics.average_precision, metrics.average_recall, metrics.loss);
}