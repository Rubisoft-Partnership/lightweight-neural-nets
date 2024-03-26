#include <stdlib.h>

#include <utils/utils.h>

// Return the number of lines in a file.
int lns(FILE *const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while ((ch = getc(file)) != EOF)
    {
        if (ch == '\n')
            lines++;
        pc = ch;
    }
    if (pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

// Read a line from a file.
char *readln(FILE *const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char *line = (char *)malloc((size) * sizeof(char));
    while ((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if (reads + 1 == size)
            line = (char *)realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// Create a matrix of doubles.
double **new2d(const int rows, const int cols)
{
    double **row = (double **)malloc((rows) * sizeof(double *));
    for (int r = 0; r < rows; r++)
        row[r] = (double *)malloc((cols) * sizeof(double));
    return row;
}
