#pragma once

void initConfusionMatrix();
void addPrediction(int true_label, int predicted_label);
void printConfusionMatrix();
void printNormalizedConfusionMatrix();