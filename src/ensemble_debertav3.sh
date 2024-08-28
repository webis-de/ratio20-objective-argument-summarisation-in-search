#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python ensemble_debertav3.py \
    --repeat 0 \
    --output ../data/argsme/ \
    --input ../data/argsme/inappropriate_arguments_sample.csv \
    --text_col argument \
    --checkpoint checkpoint-1800 \
    --model_count 5 \
    --id_col id \
    --dataset_name argsme 
