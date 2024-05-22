#ifndef METRICS_HPP
#define METRICS_HPP

extern "C"
{
#include "metrics.h"
}

#include <vector>

namespace metrics
{

    class Metrics
    {
    public:
        Metrics();
        ~Metrics() = default;

        // Generate metrics from the C library and populate the C++ class
        void generate();

        // Reset the metrics
        void reset();

        // Print the metrics
        void print() const;

        // Getters for the metrics
        float getAccuracy() const;
        float getBalancedAccuracy() const;
        float getAveragePrecision() const;
        float getAverageRecall() const;
        float getAverageF1Score() const;
        const std::vector<std::vector<float>> &getNormalizedConfusionMatrix() const;

    private:
        float accuracy;
        float balanced_accuracy;
        float average_precision;
        float average_recall;
        float average_f1_score;
        std::vector<std::vector<float>> normalized_confusion_matrix;

        // Helper function to convert C matrix to C++ matrix
        void convertConfusionMatrix(float **c_matrix, int size);
    };

} // namespace metrics

#endif // METRICS_HPP
