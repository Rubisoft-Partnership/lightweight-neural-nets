#pragma once

#include <stdio.h>

void open_log_file_with_timestamp(const char *logDir, const char *logPrefix);
void close_log_file(void);

typedef enum
{
    LOG_DEBUG, // Detailed information, typically of interest only when diagnosing problems.
    LOG_INFO,  // Informational messages that highlight the progress of the application.
    LOG_WARN,  // Potentially harmful situations.
    LOG_ERROR  // Error events that might still allow the application to continue running.
} LogLevel;

void set_log_level(LogLevel level);

void log_debug(const char *format, ...);
void log_info(const char *format, ...);
void log_warn(const char *format, ...);
void log_error(const char *format, ...);
