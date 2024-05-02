/**
 * @file metrics.h
 * @brief Header file for the metrics library.
 * 
 * The metrics include accuracy, precision, recall, and confusion matrix.
 * These metrics are used to assess the performance of prediction models.
 * 
 * @see predictions.h
 * @see confusion-matrix.h
 * @see accuracy.h
 * @see precision.h
 * @see recall.h
 */
#pragma once

typedef struct
{
    float accuracy;
    float balanced_accuracy;
    float average_precision;
    float average_recall;
    float average_f1_score;
    float** normalized_confusion_matrix;
} Metrics;

#include <predictions/predictions.h>
#include <confusion-matrix/confusion-matrix.h>
#include <accuracy/accuracy.h>
#include <precision/precision.h>
#include <recall/recall.h>
#include <f1-score/f1-score.h>

Metrics generate_metrics(void);

void reset_metrics(Metrics metrics);

void print_metrics(Metrics metrics);