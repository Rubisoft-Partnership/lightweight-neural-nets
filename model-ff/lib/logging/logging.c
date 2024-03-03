/**
 * @file logging.c
 * @brief This file contains the logging utilities.
 * */

#include <logging/logging.h>

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>

static LogLevel currentLogLevel;
static FILE *globalLogFile;

// Log functions
void log_message(LogLevel level, const char *format, va_list args);

// Function to set the current log level
void set_log_level(LogLevel level)
{
    currentLogLevel = level;
}

void open_log_file_with_timestamp(const char *logDir, const char *logPrefix)
{
    // Get the current time
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);

    // Create the log filename
    char logFilename[256];
    strftime(logFilename, sizeof(logFilename), "%Y-%m-%d_%H-%M-%S", tm_info);

    // Construct the full path
    char fullPath[512];
    snprintf(fullPath, sizeof(fullPath), "%s/%s_%s.log", logDir, logPrefix, logFilename);

    // Open the log file
    globalLogFile = fopen(fullPath, "w");
    if (!globalLogFile)
    {
        perror("Failed to open log file");
        exit(EXIT_FAILURE);
    }
}

void close_log_file(void)
{
    if (globalLogFile)
    {
        fclose(globalLogFile);
    }
}

void log_message(LogLevel level, const char *format, va_list args)
{
    if (level < currentLogLevel)
    {
        return;
    }
    if (!globalLogFile)
    {
        fprintf(stderr, "Log file is not open.\n");
        return;
    }

    const char *levelStr = "";
    switch (level)
    {
    case LOG_DEBUG:
        levelStr = "DEBUG";
        break;
    case LOG_INFO:
        levelStr = "INFO";
        break;
    case LOG_WARN:
        levelStr = "WARN";
        break;
    case LOG_ERROR:
        levelStr = "ERROR";
        break;
    }
    fprintf(globalLogFile, "[%s] ", levelStr);
    vfprintf(globalLogFile, format, args);
    fprintf(globalLogFile, "\n");
    fflush(globalLogFile);
}

void log_debug(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_DEBUG, format, args);
    va_end(args);
}

void log_info(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_INFO, format, args);
    va_end(args);
}

void log_warn(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_WARN, format, args);
    va_end(args);
}

void log_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_ERROR, format, args);
    va_end(args);
}
