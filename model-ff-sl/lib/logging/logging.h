/**
 * @file logging.h
 * @brief Header file for logging functionality.
 *
 * This file contains the declarations for logging functions and macros.
 * It provides a convenient way to log messages during program execution.
 */
#pragma once

#include <stdio.h>

#define LOGGING_LOG_PATH PROJECT_BASEPATH "/logs"

/**
 * Opens the log file with a timestamp.
 */
void open_log_file_with_timestamp(void);

/**
 * Closes the log file.
 */
void close_log_file(void);

/**
 * Enumeration of log levels.
 */
typedef enum
{
    LOG_DEBUG, // Detailed information, typically of interest only when diagnosing problems.
    LOG_INFO,  // Informational messages that highlight the progress of the application.
    LOG_WARN,  // Potentially harmful situations.
    LOG_ERROR  // Error events that might still allow the application to continue running.
} LogLevel;

/**
 * Sets the log level.
 * 
 * @param level The log level to set.
 */
void set_log_level(LogLevel level);

/**
 * Logs a debug message.
 * 
 * @param format The format string for the message.
 * @param ... The additional arguments for the format string.
 */
void log_debug(const char *format, ...);

/**
 * Logs an informational message.
 * 
 * @param format The format string for the message.
 * @param ... The additional arguments for the format string.
 */
void log_info(const char *format, ...);

/**
 * Logs a warning message.
 * 
 * @param format The format string for the message.
 * @param ... The additional arguments for the format string.
 */
void log_warn(const char *format, ...);

/**
 * Logs an error message.
 * 
 * @param format The format string for the message.
 * @param ... The additional arguments for the format string.
 */
void log_error(const char *format, ...);

/**
 * Increases the indentation level for log messages.
 */
void increase_indent(void);

/**
 * Decreases the indentation level for log messages.
 */
void decrease_indent(void);
