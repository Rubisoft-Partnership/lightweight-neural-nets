#include <iostream>

#include "tiny_dnn/tiny_dnn.h"
#include "../model-bp/src/train.cpp"


static void usage(const char *argv0)
{
    std::cout << "Usage: " << argv0 << " --data_path <path_to_dataset_folder>"
              << " --learning_rate <1>"
              << " --epochs <30>"
              << " --minibatch_size <16>"
              << " --layers_number <1>" << std::endl;
}

int main(int argc, char **argv)
{
    double learning_rate = 1;
    int epochs = 30;
    std::string data_path = "";
    int minibatch_size = 16;
    int layers_number = 1;

    if (argc == 2)
    {
        std::string argname(argv[1]);
        if (argname == "--help" || argname == "-h")
        {
            usage(argv[0]);
            return 0;
        }
    }
    for (int count = 1; count + 1 < argc; count += 2)
    {
        std::string argname(argv[count]);
        if (argname == "--learning_rate")
        {
            learning_rate = atof(argv[count + 1]);
        }
        else if (argname == "--epochs")
        {
            epochs = atoi(argv[count + 1]);
        }
        else if (argname == "--minibatch_size")
        {
            minibatch_size = atoi(argv[count + 1]);
        }
        else if (argname == "--layers_number")
        {
            layers_number = atoi(argv[count + 1]);
        }
        else if (argname == "--data_path")
        {
            data_path = std::string(argv[count + 1]);
        }
        else
        {
            std::cerr << "Invalid parameter specified - \"" << argname << "\""
                      << std::endl;
            usage(argv[0]);
            return -1;
        }
    }
    if (data_path == "")
    {
        std::cerr << "Data path not specified." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (learning_rate <= 0)
    {
        std::cerr
            << "Invalid learning rate. The learning rate must be greater than 0."
            << std::endl;
        return -1;
    }
    if (epochs <= 0)
    {
        std::cerr << "Invalid number of epochs. The number of epochs must be "
                     "greater than 0."
                  << std::endl;
        return -1;
    }
    if (layers_number <= 0)
    {
        std::cerr << "Invalid number of layers. The number of layers must be "
                     "greater than 0."
                  << std::endl;
        return -1;
    }
    if (minibatch_size <= 0 || minibatch_size > 60000)
    {
        std::cerr
            << "Invalid minibatch size. The minibatch size must be greater than 0"
               " and less than dataset size (60000)."
            << std::endl;
        return -1;
    }
    std::cout << "Running with the following parameters:" << std::endl
              << "Data path: " << data_path << std::endl
              << "Learning rate: " << learning_rate << std::endl
              << "Minibatch size: " << minibatch_size << std::endl
              << "Number of epochs: " << epochs << std::endl
              << "Number of layers: " << layers_number << std::endl
              << std::endl;
    try
    {
        train(data_path, learning_rate, epochs, minibatch_size, layers_number);
    }
    catch (tiny_dnn::nn_error &err)
    {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
    return 0;
}
