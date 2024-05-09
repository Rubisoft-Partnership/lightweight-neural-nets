#include <metrics.h>

#include <stdlib.h>
#include <stdio.h>

Metrics generate_metrics(void)
{
    Metrics metrics;
    metrics.accuracy = get_accuracy();
    metrics.average_f1_score = get_average_f1_score();
    metrics.average_precision = get_average_precision();
    metrics.average_recall = get_average_recall();
    metrics.balanced_accuracy = get_balanced_accuracy();
    metrics.normalized_confusion_matrix = get_normalized_confusion_matrix();
    return metrics;
}

void print_metrics(Metrics metrics)
{
    if (metrics.normalized_confusion_matrix == NULL)
    {
        printf("No metrics to print.\n");
        return;
    }
    printf("Accuracy: %f\n", metrics.accuracy);
    printf("Average F1 Score: %f\n", metrics.average_f1_score);
    printf("Average Precision: %f\n", metrics.average_precision);
    printf("Average Recall: %f\n", metrics.average_recall);
    printf("Balanced Accuracy: %f\n", metrics.balanced_accuracy);
    printf("Normalized Confusion Matrix:\n");
    print_normalized_confusion_matrix();
}

void reset_metrics(Metrics metrics)
{
    metrics.accuracy = 0;
    metrics.average_f1_score = 0;
    metrics.average_precision = 0;
    metrics.average_recall = 0;
    metrics.balanced_accuracy = 0;
    free(metrics.normalized_confusion_matrix);
    metrics.normalized_confusion_matrix = NULL;
}