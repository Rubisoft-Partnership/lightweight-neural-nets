/**
 * @file logging.c
 * @brief This file contains the logging utilities.
 * */

#include <logging/logging.h>

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <sys/stat.h>

/**
 * @brief The current log level for logging messages.
 */
static LogLevel currentLogLevel;

/**
 * @brief The global log file pointer.
 */
static FILE *globalLogFile;

/**
 * @brief The current indentation level for log messages.
 */
static int indentLevel = 0;

/**
 * Logs a message with the specified log level.
 *
 * @param level The log level of the message.
 * @param format The format string for the message.
 * @param args The variable arguments list for the format string.
 */
void log_message(LogLevel level, const char *format, va_list args);

/**
 * Sets the current log level.
 *
 * This function allows you to set the log level for the logging system.
 * The log level determines which log messages will be displayed.
 *
 * @param level The log level to set.
 */
void set_log_level(LogLevel level)
{
    currentLogLevel = level;
}

/**
 * @brief Open a log file with a timestamp.
 *
 * The log file is created with a filename in the format "log_<timestamp>.log".
 * The log file is opened in write mode.
 * If the log file fails to open, an error message is printed and the program exits.
 *
 * @param None
 */
void open_log_file_with_timestamp(void)
{
    // Get the current time
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);

    // Create the log filename
    char logFilename[256];
    strftime(logFilename, sizeof(logFilename), "%Y-%m-%d_%H-%M-%S", tm_info);

    // Construct the full path
    char fullPath[256];
    snprintf(fullPath, sizeof(fullPath), "%s/log_%s.log", LOGGING_LOG_PATH, logFilename);

    // Create the log directory if it doesn't exist
    mkdir(LOGGING_LOG_PATH, 0777);

    // Open the log file
    globalLogFile = fopen(fullPath, "w");
    if (!globalLogFile)
    {
        perror("Failed to open log file");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Closes the log file if it is open.
 *
 * This function checks if the globalLogFile is not NULL and closes the file using the fclose function.
 *
 * @param None
 */
void close_log_file(void)
{
    if (globalLogFile)
    {
        fclose(globalLogFile);
    }
}
/**
 * Logs a message with the specified log level.
 *
 * @param level The log level of the message.
 * @param format The format string for the message.
 * @param args The variable arguments list for the format string.
 */
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
    for (int i = 0; i < indentLevel; i++)
    {
        fprintf(globalLogFile, "\t");
    }
    fprintf(globalLogFile, "[%s] ", levelStr);
    vfprintf(globalLogFile, format, args);
    fprintf(globalLogFile, "\n");
    fflush(globalLogFile);
}

/**
 * Logs a debug message with the specified format and arguments.
 *
 * @param format The format string for the debug message.
 * @param ... The variable number of arguments to be formatted and logged.
 */
void log_debug(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_DEBUG, format, args);
    va_end(args);
}

/**
 * Logs an informational message.
 *
 * @param format The format string for the message.
 * @param ... The variable arguments to be formatted and logged.
 */
void log_info(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_INFO, format, args);
    va_end(args);
}

/**
 * Logs a warning message.
 *
 * @param format The format string for the message.
 * @param ... The variable arguments to be formatted and logged.
 */
void log_warn(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_WARN, format, args);
    va_end(args);
}

/**
 * Logs an error message.
 *
 * @param format The format string for the message.
 * @param ... The variable arguments to be formatted and logged.
 */
void log_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    log_message(LOG_ERROR, format, args);
    va_end(args);
}

/**
 * @brief Increases the current indentation level.
 *
 * This function is used to increase the current indentation level. It increments the value of the `indentLevel` variable by 1.
 *
 * @param None
 */
void increase_indent(void)
{
    indentLevel++;
}

/**
 * @brief Decreases the current indentation level.
 *
 * This function is used to decrease the current indentation level. It decrements the value of the `indentLevel` variable by 1.
 *
 * @param None
 */
void decrease_indent(void)
{
    indentLevel--;
}
