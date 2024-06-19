#!/usr/bin/env python3

from simple_term_menu import TerminalMenu
from sklearn import datasets
import os
import numpy as np
from sklearn.model_selection import train_test_split
import idx2numpy

DATASETS_ALL = 0
DATASETS_DIGITS = 1
DATASETS_MNIST = 2

MODEL_ALL = 0
MODEL_FF = 1
MODEL_BP = 2


PATH_DIGITS = "digits/"
PATH_MNIST = "mnist/"

GLOBAL_DATASET = "/global/"
CLIENT_DATASET_PREFIX = "/client-"

TEST_DATASET_PERCENTAGE = 0.2

BINARY_MNIST_BASEPATH = "../../tiny-dnn/data/"

ACTION_GENERATE = 0
ACTION_DELETE = 1


def main():
    # Check if mnist and digits folders exist
    if not os.path.exists(PATH_DIGITS):
        os.makedirs(PATH_DIGITS)
    if not os.path.exists(PATH_MNIST):
        os.makedirs(PATH_MNIST)

    # Default values
    selected_model = MODEL_ALL
    seleted_dataset = DATASETS_ALL
    number_of_datasets = 10
    selected_action = ACTION_GENERATE

    # Check if there are cli arguments
    if len(os.sys.argv) > 5:
        print("Too many arguments")
        os.sys.exit(1)
    if len(os.sys.argv) > 1:
        if os.sys.argv[1] == "help":
            print(
                "Usage: python3 generate_datasets.py [action] [dataset] [model] [number_of_datasets]")
            print("action: gen | del - default: gen")
            print("dataset: all | digits | mnist - default: all")
            print("model: all | ff | bp - default: all")
            print("number_of_datasets: int - default: {}".format(number_of_datasets))
            os.sys.exit(0)
        if os.sys.argv[1] == "gen":
            selected_action = ACTION_GENERATE
        elif os.sys.argv[1] == "del":
            selected_action = ACTION_DELETE
        else:
            print("Invalid action. Use 'gen' or 'del'")
            os.sys.exit(1)
        if len(os.sys.argv) > 2:
            if os.sys.argv[2] == "all":
                seleted_dataset = DATASETS_ALL
            elif os.sys.argv[2] == "digits":
                seleted_dataset = DATASETS_DIGITS
            elif os.sys.argv[2] == "mnist":
                seleted_dataset = DATASETS_MNIST
            else:
                print("Invalid dataset. Use 'all', 'digits' or 'mnist'")
                os.sys.exit(1)
        if len(os.sys.argv) > 3:
            if os.sys.argv[3] == "all":
                selected_model = MODEL_ALL
            elif os.sys.argv[3] == "ff":
                selected_model = MODEL_FF
            elif os.sys.argv[3] == "bp":
                selected_model = MODEL_BP
            else:
                print("Invalid model. Use 'all', 'ff' or 'bp'")
                os.sys.exit(1)
        if len(os.sys.argv) > 4:
            number_of_datasets = int(os.sys.argv[4])
            if number_of_datasets < 1:
                print("Invalid number of datasets. Must be greater than 0")
                os.sys.exit(1)
    else:
        dataset_actions = ["Generate federated dataset",
                           "Delete federated dataset"]
        terminal_menu = TerminalMenu(
            dataset_actions, title="Select an action:")
        selected_action = terminal_menu.show()
        seleted_dataset = select_dataset()
        if selected_action == ACTION_GENERATE:
            selected_model = select_model()
            number_of_datasets = select_datasets_number()

    if selected_action == ACTION_GENERATE:
        generate_federated_datasets(
            selected_model, seleted_dataset, number_of_datasets)
    elif selected_action == ACTION_DELETE:
        delete_federated_dataset(seleted_dataset)


def generate_federated_datasets(model: int = MODEL_ALL, dataset: int = DATASETS_ALL, number_of_datasets: int = 10):
    if dataset == DATASETS_ALL or dataset == DATASETS_DIGITS:
        generate_digits_datasets(model, number_of_datasets)
    if dataset == DATASETS_ALL or dataset == DATASETS_MNIST:
        generate_mnist_datasets(model, number_of_datasets)


def generate_mnist_datasets(selected_model: int, number_of_datasets: int):
    create_folders(number_of_datasets, PATH_MNIST)
    train_images = idx2numpy.convert_from_file(
        BINARY_MNIST_BASEPATH + "train-images.idx3-ubyte")
    train_labels = idx2numpy.convert_from_file(
        BINARY_MNIST_BASEPATH + "train-labels.idx1-ubyte")
    test_images = idx2numpy.convert_from_file(BINARY_MNIST_BASEPATH +
                                              "t10k-images.idx3-ubyte")
    test_labels = idx2numpy.convert_from_file(BINARY_MNIST_BASEPATH +
                                              "t10k-labels.idx1-ubyte")
    if selected_model == MODEL_BP or selected_model == MODEL_ALL:
        train_images_split = np.array_split(
            train_images, number_of_datasets - 1)
        train_labels_split = np.array_split(
            train_labels, number_of_datasets - 1)
        test_images_split = np.array_split(test_images, number_of_datasets)
        test_labels_split = np.array_split(test_labels, number_of_datasets)

        for dataset in range(number_of_datasets - 1):
            idx2numpy.convert_to_file(PATH_MNIST + CLIENT_DATASET_PREFIX + str(dataset) + "/train-images.idx3-ubyte",
                                      train_images_split[dataset])
            idx2numpy.convert_to_file(PATH_MNIST + CLIENT_DATASET_PREFIX + str(dataset) + "/train-labels.idx1-ubyte",
                                      train_labels_split[dataset])
            idx2numpy.convert_to_file(PATH_MNIST + CLIENT_DATASET_PREFIX + str(dataset) + "/t10k-images.idx3-ubyte",
                                      test_images_split[dataset])
            idx2numpy.convert_to_file(PATH_MNIST + CLIENT_DATASET_PREFIX + str(dataset) + "/t10k-labels.idx1-ubyte",
                                      test_labels_split[dataset])

        idx2numpy.convert_to_file(PATH_MNIST + GLOBAL_DATASET + "/t10k-images.idx3-ubyte",
                                  test_images_split[number_of_datasets - 1])
        idx2numpy.convert_to_file(PATH_MNIST + GLOBAL_DATASET + "/t10k-labels.idx1-ubyte",
                                  test_labels_split[number_of_datasets - 1])
        print("Generated MNIST datasets for model BP")
    if selected_model == MODEL_FF or selected_model == MODEL_ALL:
        # Flatten the images
        train_images = train_images.reshape(-1, 28 * 28)
        test_images = test_images.reshape(-1, 28 * 28)

        train_images_split = np.array_split(
            train_images, number_of_datasets - 1)
        train_labels_split = np.array_split(
            train_labels, number_of_datasets - 1)
        test_images_split = np.array_split(test_images, number_of_datasets)
        test_labels_split = np.array_split(test_labels, number_of_datasets)

        for dataset in range(number_of_datasets - 1):
            with open(PATH_MNIST + CLIENT_DATASET_PREFIX + str(dataset) + "/train.txt", "w") as f:
                for row in range(len(train_images_split[dataset])):
                    f.write(
                        " ".join(map(str, train_images_split[dataset][row])) + " ")
                    one_hot_target = np.zeros(10)
                    one_hot_target[train_labels_split[dataset]] = 1
                    f.write(" ".join(map(str, one_hot_target)) + "\n")
            with open(PATH_MNIST + CLIENT_DATASET_PREFIX + str(dataset) + "/test.txt", "w") as f:
                for row in range(len(test_images_split[dataset])):
                    f.write(
                        " ".join(map(str, test_images_split[dataset][row])) + " ")
                    one_hot_target = np.zeros(10)
                    one_hot_target[test_labels_split[dataset]] = 1
                    f.write(" ".join(map(str, one_hot_target)) + "\n")
        # Test dataset for global model
        with open(PATH_MNIST + GLOBAL_DATASET + "/test.txt", "w") as f:
            for row in range(len(test_images_split[number_of_datasets - 1])):
                f.write(
                    " ".join(map(str, test_images_split[number_of_datasets - 1][row])) + " ")
                one_hot_target = np.zeros(10)
                one_hot_target[test_labels_split[number_of_datasets - 1]] = 1
                f.write(" ".join(map(str, one_hot_target)) + "\n")
        print("Generated MNIST datasets for model FF")


def generate_digits_datasets(selected_model: int, number_of_datasets: int):
    if selected_model == MODEL_FF or selected_model == MODEL_ALL:
        joined_digits = get_joined_digits()
        samples_per_dataset = len(joined_digits) // number_of_datasets
        create_folders(number_of_datasets, PATH_DIGITS)
        for i in range(number_of_datasets - 1):
            train, test = train_test_split(joined_digits[i * samples_per_dataset:(i + 1) * samples_per_dataset],
                                           test_size=TEST_DATASET_PERCENTAGE, random_state=42)
            with open(PATH_DIGITS + CLIENT_DATASET_PREFIX + str(i) + "/train.txt", "w") as f:
                for line in train:
                    f.write(" ".join(map(str, line)) + "\n")
            with open(PATH_DIGITS + CLIENT_DATASET_PREFIX + str(i) + "/test.txt", "w") as f:
                for line in test:
                    f.write(" ".join(map(str, line)) + "\n")
        # Write only test dataset for global model
        with open(PATH_DIGITS + GLOBAL_DATASET + "/test.txt", "w") as f:
            global_test = joined_digits[(
                number_of_datasets - 1) * samples_per_dataset:]
            for line in global_test:
                f.write(" ".join(map(str, line)) + "\n")
        print("Generated Digits datasets for model FF")


def get_joined_digits():
    (digits_feat, digits_gt) = datasets.load_digits(return_X_y=True)
    digits_one_hot_targets = np.zeros((len(digits_gt), 10))
    digits_one_hot_targets[np.arange(len(digits_gt)), digits_gt] = 1
    zeros_padding = np.zeros((len(digits_gt), 10))
    joined_digits = np.concatenate(
        (digits_feat, zeros_padding, digits_one_hot_targets), axis=1)
    # Shuffle the dataset
    np.random.shuffle(joined_digits)
    return joined_digits


def create_folders(number_of_datasets: int, base_path: str):
    for i in range(number_of_datasets - 1):
        full_path = base_path + CLIENT_DATASET_PREFIX + str(i)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
    full_path = base_path + GLOBAL_DATASET
    if not os.path.exists(full_path):
        os.makedirs(full_path)


def select_datasets_number():
    number_of_datasets = 10
    print("Insert the number of datasets you want to generate [default: {}]:".format(
        number_of_datasets))
    input_number = input()
    if input_number:
        number_of_datasets = int(input_number)
    return number_of_datasets


def delete_federated_dataset(dataset: int):
    if dataset == DATASETS_ALL or dataset == DATASETS_DIGITS:
        # rm -rf digits/*
        os.system("rm -rf digits/*")
        print("Deleted Digits dataset")
    if dataset == DATASETS_ALL or dataset == DATASETS_MNIST:
        # rm -rf mnist/*
        os.system("rm -rf mnist/*")
        print("Deleted MNIST dataset")


def select_dataset() -> int:
    datasets = ["All", "Digits", "MNIST"]
    terminal_menu = TerminalMenu(datasets, title="Select a dataset:")
    menu_entry_index = terminal_menu.show()
    return menu_entry_index


def select_model() -> int:
    models = ["All", "ForwardForward", "Backpropagation"]
    terminal_menu = TerminalMenu(models, title="Select a model:")
    menu_entry_index = terminal_menu.show()
    return menu_entry_index


if __name__ == "__main__":
    main()
