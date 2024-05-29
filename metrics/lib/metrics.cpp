#include "metrics.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

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
        convertConfusionMatrix(c_metrics.normalized_confusion_matrix, NUM_CLASSES);
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
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << val << " ";
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
            ss << std::setw(6) << std::fixed << std::setprecision(2) << val << " ";
            }
            ss << std::endl;
        }
        return ss.str();
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

    Metrics mean(const std::vector<Metrics> &metrics)
    {
        Metrics mean_metrics;
        for (const auto &metric : metrics)
        {
            mean_metrics.accuracy += metric.accuracy;
            mean_metrics.balanced_accuracy += metric.balanced_accuracy;
            mean_metrics.average_precision += metric.average_precision;
            mean_metrics.average_recall += metric.average_recall;
            mean_metrics.average_f1_score += metric.average_f1_score;

            // Accumulate the confusion matrix
            const auto &confusion_matrix = metric.normalized_confusion_matrix;
            if (mean_metrics.normalized_confusion_matrix.empty())
                mean_metrics.normalized_confusion_matrix = confusion_matrix;
            else
                for (size_t i = 0; i < confusion_matrix.size(); ++i)
                    for (size_t j = 0; j < confusion_matrix[i].size(); ++j)
                        mean_metrics.normalized_confusion_matrix[i][j] += confusion_matrix[i][j];
        }

        // Divide by the number of metrics to get the mean
        mean_metrics.accuracy /= metrics.size();
        mean_metrics.balanced_accuracy /= metrics.size();
        mean_metrics.average_precision /= metrics.size();
        mean_metrics.average_recall /= metrics.size();
        mean_metrics.average_f1_score /= metrics.size();
        for (size_t i = 0; i < mean_metrics.normalized_confusion_matrix.size(); ++i)
            for (size_t j = 0; j < mean_metrics.normalized_confusion_matrix[i].size(); ++j)
                mean_metrics.normalized_confusion_matrix[i][j] /= metrics.size();
        return mean_metrics;
    }

} // namespace metrics
