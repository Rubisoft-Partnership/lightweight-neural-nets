#include <config/config.hpp>

#include <spdlog/spdlog.h>

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
void init_metrics_logger();

namespace config
{
    std::string basepath;
    std::string datasets_path;
    std::string simulation_path;
    std::string checkpoints_path;
    std::string log_path;
    std::string simulation_timestamp;
    std::string selected_dataset = dataset_mnist;

    ModelType model_type = ModelType::BP;

    namespace training
    {
        float learning_rate = 0.01;
        int batch_size = 32;
        int epochs = 5;
    }

    namespace orchestrator
    {
        size_t num_clients = 10;
        size_t num_rounds = 3;
        float c_rate = 0.1;
        float checkpoint_rate = 0.2;
    }

    namespace parameters
    {
        int num_classes = 10;
        std::vector<int> units = {784, 100, 10};
        namespace ff
        {
            float threshold = 5.0;
            float beta1 = 0.9;
            float beta2 = 0.999;
            LossType loss = LossType::LOSS_TYPE_FF;
        }
    }
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

void config::log_simulation_params()
{
    spdlog::info("Logging simulation parameters:");
    spdlog::info("Orchestration parameters:");
    spdlog::info("Number of clients: {}", orchestrator::num_clients);
    spdlog::info("Number of rounds: {}", orchestrator::num_rounds);
    spdlog::info("Client selection rate: {}", orchestrator::c_rate);
    spdlog::info("Checkpoint rate: {}", orchestrator::checkpoint_rate);
    spdlog::info("Training parameters:");
    spdlog::info("Learning rate: {}", training::learning_rate);
    spdlog::info("Batch size: {}", training::batch_size);
    spdlog::info("Epochs: {}", training::epochs);
    spdlog::info("Model parameters:");
    switch (model_type)
    {
    case config::ModelType::FF:
        spdlog::info("FF model");
        spdlog::info("Threshold: {}", parameters::ff::threshold);
        spdlog::info("Beta1: {}", parameters::ff::beta1);
        spdlog::info("Beta2: {}", parameters::ff::beta2);
        switch (parameters::ff::loss)
        {
        case LossType::LOSS_TYPE_FF:
            spdlog::info("FF loss");
            break;
        case LossType::LOSS_TYPE_SYMBA:
            spdlog::info("SymBa loss");
            break;
        default:
            spdlog::error("Unknown loss type while logging simulation parameters");
            break;
        }
        break;
    case config::ModelType::BP:
        spdlog::info("BP model");
        break;
    default:
        spdlog::error("Unknown model type while logging simulation parameters");
        break;
    }
    spdlog::info("Units per layer: {}", [&]()
                 {
        std::string units_str = "[ ";
        for (int unit : parameters::units)
            units_str += std::to_string(unit) + " ";
        units_str += "]";
        return units_str; }());
    spdlog::info("Number of classes: {}", parameters::num_classes);
    spdlog::info("Finished logging simulation parameters\n");
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