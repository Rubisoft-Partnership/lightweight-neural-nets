#pragma once

#include <stdio.h>

int lns(FILE *const file);
char *readln(FILE *const file);
double **new2d(const int rows, const int cols);
void set_seed(const int s);
int get_random(void);
