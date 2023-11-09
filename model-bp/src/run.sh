#!/bin/bash

filename=${1:-main.out}
data_path=${2:-../../tiny-dnn/data}
learning_rate=${3:-1}
epochs=${4:-30}
minibatch_size=${5:-16}
layers_number=${6:-1}

if [ ! -f $filename ]; then
    make
fi

./$filename --data_path $data_path --learning_rate $learning_rate --epochs $epochs --minibatch_size $minibatch_size --layers_number $layers_number