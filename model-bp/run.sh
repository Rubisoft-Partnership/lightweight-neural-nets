#!/bin/bash

filename=${1:-target/main.out}
data_path=${2:-../tiny-dnn/data}
learning_rate=${3:-0.01}
epochs=${4:-30}
minibatch_size=${5:-32}

if [ ! -f $filename ]; then
    make
fi

./$filename --data_path $data_path --learning_rate $learning_rate --epochs $epochs --minibatch_size $minibatch_size --layer_units 784 500 500 500 10