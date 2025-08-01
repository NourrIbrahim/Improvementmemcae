#!/bin/bash

echo $*

dataset=$1
dev_eval=$2
id_1=$3
id_2=$4
id_3=$5
id_4=$6

id="$id_1 $id_2 $id_3 $id_4"

echo dataset ${dataset}
echo dev_eval ${dev_eval}
echo id ${id}

tag="id("
for i in ${id}; do
    tag="${tag}${i}_"
done
tag="${tag})"

# Ensure Python is found
export PATH="/c/Python312:$PATH"

# Run Python training script using full path
/c/Python312/python train.py \
    --dataset=${dataset} \
    ${dev_eval} \
    -tag=${tag} \
    --use_ids ${id} \
    --train_only
