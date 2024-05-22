#include "metrics.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace metrics
{

    Metrics::Metrics()
        : accuracy(0), balanced_accuracy(0), average_precision(0),
          average_recall(0), average_f1_score(0)
    {
    }

    void Metrics::generate()
    {
        ::Metrics c_metrics = generate_metrics();
        accuracy = c_metrics.accuracy;
        balanced_accuracy = c_metrics.balanced_accuracy;
        average_precision = c_metrics.average_precision;
        average_recall = c_metrics.average_recall;
        average_f1_score = c_metrics.average_f1_score;
        // Assuming size is known or provided, replace 'size' with actual size
        convertConfusionMatrix(c_metrics.normalized_confusion_matrix, NUM_CLASSES * NUM_CLASSES);
        // Free the C matrix as it's no longer needed
        free(c_metrics.normalized_confusion_matrix);
    }

    void Metrics::reset()
    {
        accuracy = 0;
        balanced_accuracy = 0;
        average_precision = 0;
        average_recall = 0;
        average_f1_score = 0;
        normalized_confusion_matrix.clear();
    }

    void Metrics::print() const
    {
        std::cout << "Accuracy: " << accuracy << std::endl;
        std::cout << "Average F1 Score: " << average_f1_score << std::endl;
        std::cout << "Average Precision: " << average_precision << std::endl;
        std::cout << "Average Recall: " << average_recall << std::endl;
        std::cout << "Balanced Accuracy: " << balanced_accuracy << std::endl;
        std::cout << "Normalized Confusion Matrix:" << std::endl;
        for (const auto &row : normalized_confusion_matrix)
        {
            for (const auto &val : row)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    std::string Metrics::toString() const
    {
        std::stringstream ss;
        ss << "Accuracy: " << accuracy << std::endl;
        ss << "Average F1 Score: " << average_f1_score << std::endl;
        ss << "Average Precision: " << average_precision << std::endl;
        ss << "Average Recall: " << average_recall << std::endl;
        ss << "Balanced Accuracy: " << balanced_accuracy << std::endl;
        ss << "Normalized Confusion Matrix:" << std::endl;
        for (const auto &row : normalized_confusion_matrix)
        {
            for (const auto &val : row)
            {
                ss << val << " ";
            }
            ss << std::endl;
        }
        return ss.str();
    }

    float Metrics::getAccuracy() const
    {
        return accuracy;
    }

    float Metrics::getBalancedAccuracy() const
    {
        return balanced_accuracy;
    }

    float Metrics::getAveragePrecision() const
    {
        return average_precision;
    }

    float Metrics::getAverageRecall() const
    {
        return average_recall;
    }

    float Metrics::getAverageF1Score() const
    {
        return average_f1_score;
    }

    const std::vector<std::vector<float>> &Metrics::getNormalizedConfusionMatrix() const
    {
        return normalized_confusion_matrix;
    }

    void Metrics::convertConfusionMatrix(float **c_matrix, int size)
    {
        normalized_confusion_matrix.resize(size);
        for (int i = 0; i < size; ++i)
        {
            normalized_confusion_matrix[i].resize(size);
            for (int j = 0; j < size; ++j)
            {
                normalized_confusion_matrix[i][j] = c_matrix[i][j];
            }
        }
    }

} // namespace metrics
