#pragma once

void initConfusionMatrix(void);
void addPrediction(int true_label, int predicted_label);
void printConfusionMatrix(void);
void printNormalizedConfusionMatrix(void);
