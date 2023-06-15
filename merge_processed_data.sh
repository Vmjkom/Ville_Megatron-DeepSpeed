#!/bin/bash

DIR=$(basename -s .bin data/bert/finbert_data/*.bin)



python3 /scratch/project_2004600/ville/BigScience/Megatron-DeepSpeed/merge_preprocessed_data.py \
    --datasets $(cat paths.txt) \
    --output-prefix finbert_train
