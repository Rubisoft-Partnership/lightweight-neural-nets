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
        float accuracy;
        float balanced_accuracy;
        float average_precision;
        float average_recall;
        float average_f1_score;
        std::vector<std::vector<float>> normalized_confusion_matrix;

        Metrics();
        ~Metrics() = default;

        // Generate metrics from the C library and populate the C++ class
        void generate();

        // Reset the metrics
        void reset();

        // Print the metrics
        void print() const;
        // To string representation
        std::string toString() const;

    private:
        // Helper function to convert C matrix to C++ matrix
        void convertConfusionMatrix(float **c_matrix, int size);
    };

    // Function to compute the mean of a vector of Metrics objects
    Metrics mean(const std::vector<Metrics> &metrics);

} // namespace metrics

#endif // METRICS_HPP
