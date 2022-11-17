#!/bin/bash

BS_DIR=/pfs/lustrep4/scratch/project_462000119/ville/BS_Megatron/tools
DATA_DIR=/pfs/lustrep4/scratch/project_462000119/data/FinBERT/jsonl/train/*



for f in $DATA_DIR
do
       python3 tools/preprocess_data.py \
              --input data/owt2-2020-01.jsonl \
              --output-prefix data/bert/finbert_data/owt2-2020-01 \
              --vocab bert/bert-base-cased-vocab.txt \
              --dataset-impl mmap \
              --tokenizer-type BertWordPieceCase \
              --split-sentences \
              --workers $SLURM_CPUS_PER_TASK
done

