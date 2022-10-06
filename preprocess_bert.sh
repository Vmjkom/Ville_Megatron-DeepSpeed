#!/bin/bash

python tools/preprocess_data.py \
       --input data/tiny-owt2-sample.jsonl \
       --output-prefix tiny-owt2-sample \
       --vocab nvidia/megatron-bert-cased-345m/vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceCase \
       --split-sentences \
       --workers $SLURM_CPUS_PER_TASK