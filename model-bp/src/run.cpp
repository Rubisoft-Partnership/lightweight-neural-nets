#include <iostream>
#include <iomanip>

#include <tiny_dnn/tiny_dnn.h>
#include <../model-bp/src/train.cpp>

void usage(char *argv0, double learning_rate, int epochs, std::vector<int> layer_units, int batch_size, std::string data_path)
{
    std::cout << "Usage: " << argv0 << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -lr, --learning_rate\tLearning rate for the optimizer (default: " << learning_rate << ")" << std::endl;
    std::cout << "  -e,  --epochs\t\tNumber of epochs for training (default: " << epochs << ")" << std::endl;
    std::cout << "  -bs, --batch_size\tBatch size for training (default: " << batch_size << ")" << std::endl;
    std::cout << "  -lu, --layer_units\tWidth of each layer (default: ";
    for (int i = 0; i < layer_units.size(); i++)
    {
        std::cout << layer_units[i] << " ";
    }
    std::cout << ")" << std::endl;
    std::cout << "  -dp, --dataset_path\tPath to the dataset (default: " << data_path << ")" << std::endl;
}

int main(int argc, char **argv)
{
    double learning_rate = 0.01;
    int epochs = 30;
    std::string data_path = "../tiny-dnn/data";
    int batch_size = 16;
    std::vector<int> layer_units = std::vector<int>();

    if (argc == 2 && (argv[1] == "--help" || argv[1] == "-h"))
    {
        usage(argv[0], learning_rate, epochs, layer_units, batch_size, data_path);
        exit(0);
    }
    for (int count = 1; count + 1 < argc; count += 2)
    {
        std::string argname(argv[count]);
        if (argname == "--learning_rate" || argname == "-lr")
        {
            learning_rate = atof(argv[count + 1]);
        }
        else if (argname == "--epochs" || argname == "-e")
        {
            epochs = atoi(argv[count + 1]);
        }
        else if (argname == "--batch_size" || argname == "-bs")
        {
            batch_size = atoi(argv[count + 1]);
        }
        else if (argname == "--layer_units" || argname == "-lu")
        {
            // proceed to read a list of integers and store them in layer_units
            // if atoi fails, then there are no more integers to read
            count++;
            int read_int = atoi(argv[count]);
            while (read_int)
            {
                layer_units.push_back(read_int);
                count++;
                if (count < argc)
                    read_int = atoi(argv[count]);
                else
                    read_int = 0;
            }
            count -= 2;
        }
        else if (argname == "--dataset_path" || argname == "-dp")
        {
            data_path = std::string(argv[count + 1]);
        }
        else
        {
            usage(argv[0], learning_rate, epochs, layer_units, batch_size, data_path);
            exit(-1);
        }
    }
    if (data_path == "")
    {
        std::cerr << "Data path not specified." << std::endl;
        usage(argv[0], learning_rate, epochs, layer_units, batch_size, data_path);
        exit(-1);
    }
    if (learning_rate <= 0)
    {
        std::cerr
            << "Invalid learning rate. The learning rate must be greater than 0."
            << std::endl;
        exit(-1);
    }
    if (epochs <= 0)
    {
        std::cerr << "Invalid number of epochs. The number of epochs must be "
                     "greater than 0."
                  << std::endl;
        exit(-1);
    }
    if (layer_units.size() <= 1)
    {
        std::cerr << "Invalid number of units. The number of layers must be "
                     "greater than 1."
                  << std::endl;
        exit(-1);
    }
    if (batch_size <= 0 || batch_size > 60000)
    {
        std::cerr
            << "Invalid minibatch size. The minibatch size must be greater than 0"
               " and less than dataset size (60000)."
            << std::endl;
        exit(-1);
    }
    std::cout << "Running with the following parameters:" << std::endl
              << "Data path: " << data_path << std::endl
              << "Learning rate: " << learning_rate << std::endl
              << "Minibatch size: " << batch_size << std::endl
              << "Number of epochs: " << epochs << std::endl
              << "Number of layers: " << layer_units.size() - 1 << std::endl;
    // prints the layers and its in and out dim using iomanip

#define PRINT_WIDTH 12

    std::cout << std::endl
              << std::left
              << std::setw(PRINT_WIDTH) << "Layer num"
              << std::setw(PRINT_WIDTH) << "Input dim"
              << std::setw(PRINT_WIDTH) << "Output dim"
              << std::endl;

    for (int i = 0; i < layer_units.size() - 1; i++)
    {
        std::cout << std::left
                  << std::setw(PRINT_WIDTH) << i + 1
                  << std::setw(PRINT_WIDTH) << layer_units[i]     // Replace with actual in_dim
                  << std::setw(PRINT_WIDTH) << layer_units[i + 1] // Replace with actual out_dim
                  << std::endl;
    }
    std::cout << std::endl;

    try
    {
        train(data_path, learning_rate, epochs, batch_size, layer_units);
    }
    catch (tiny_dnn::nn_error &err)
    {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
    return 0;
}
