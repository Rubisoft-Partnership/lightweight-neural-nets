#include <config/config.hpp>

#include <chrono>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <unistd.h>
#include <limits.h>
using namespace config;

static std::string get_timestamp();
static int find_first_available_folder(const std::string &base_path);
std::string getExecutableFullpath();
std::string getExecutableBasepath();

namespace config
{
    std::string basepath;
    std::string datasets_path;
    std::string simulation_path;
    std::string checkpoints_path;
    std::string log_path;
    std::string simulation_timestamp;

    int num_classes = 10;
    ModelType model_type = ModelType::BP;
}

void config::init_config()
{
    // Locate project base path. The executable is located in /framework/target/.
    basepath = getExecutableBasepath() + "/../../";

    datasets_path = basepath + datasets_folder;

    std::string folder_num = std::to_string(find_first_available_folder(basepath + simulations_folder));
    simulation_path = basepath + simulations_folder + folder_num + "/";

    checkpoints_path = simulation_path + checkpoints_folder;

    simulation_timestamp = get_timestamp();
    log_path = basepath + logs_folder + folder_num + "_" + simulation_timestamp + ".log";

    // Default parameters
    model_bp_parameters.units = {784, 100, 10};
}

static std::string get_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto itt = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(gmtime(&itt), "%Y%m%d%H%M%S");
    return ss.str();
}

static int find_first_available_folder(const std::string &base_path)
{
    int folder_number = 1;
    while (true)
    {
        std::string folder_path = base_path + std::to_string(folder_number);
        if (!std::filesystem::exists(folder_path))
        {
            return folder_number;
        }
        ++folder_number;
    }
}

#if defined(_WIN32)
#include <windows.h>
std::string getExecutableFullpath()
{
    char result[MAX_PATH];
    GetModuleFileName(NULL, result, MAX_PATH);
    return std::string(result);
}
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
std::string getExecutableFullpath()
{
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <limits.h>
std::string getExecutableFullpath()
{
    char result[PATH_MAX];
    uint32_t size = sizeof(result);
    if (_NSGetExecutablePath(result, &size) == 0)
    {
        return std::string(result);
    }
    else
    {
        // Buffer too small; resize and try again
        char *dynamicResult = new char[size];
        _NSGetExecutablePath(dynamicResult, &size);
        std::string path(dynamicResult);
        delete[] dynamicResult;
        return path;
    }
}
#else
std::string getExecutableFullpath()
{
    // Unsupported platform
    return "";
}
#endif

std::string getExecutableBasepath()
{
    std::string exePath = getExecutableFullpath();
    if (exePath.empty())
    {
        return "";
    }

    std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
    return exeDir.string();
}