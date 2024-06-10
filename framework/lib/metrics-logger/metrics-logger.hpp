#ifndef METRICS_LOGGER_HPP
#define METRICS_LOGGER_HPP

#include <metrics.hpp>

void init_metrics_logger();

enum DatasetType
{
    GLOBAL = 0,
    LOCAL = 1,
};

void log_metrics(const int round_num, const int client_id, const int epoch, const DatasetType dataset_type, const metrics::Metrics &metrics);

#endif // METRICS_LOGGER_HPP