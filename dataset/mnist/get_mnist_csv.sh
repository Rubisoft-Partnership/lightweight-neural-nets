#!/bin/bash

out_folder=${1:-dataset/mnist}
base_url="https://raw.githubusercontent.com/phoebetronic/mnist/main/"
train_file="mnist_train.csv"
test_file="mnist_test.csv"


if [ ! -d $out_folder ]; then
    echo "No such directory: $out_folder"
    exit 1
fi

wget -P $out_folder $base_url/$train_file.zip
wget -P $out_folder $base_url/$test_file.zip

unzip $out_folder/$train_file.zip -d $out_folder
unzip $out_folder/$test_file.zip -d $out_folder

rm $out_folder/$train_file.zip $out_folder/$test_file.zip
rm -rf $out_folder/__MACOSX
