#!/bin/bash
TENSOR_MODEL_PARALLEL_SIZE=2
TARGET_PIPELINE_MODEL_PARALLEL_SIZE=2

VOCAB_FILE=bert/finbert_vocab.txt
CHECKPOINT_PATH=checkpoints/bert/3.9B/Bert3.9b_400gpu_bigrun/global_step130000
WORLD_SIZE=2 

python3 tools/merge_mp_partitions.py \
        --model-type BERT \
        --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
        --pipeline-model-parallel-size 1 \
        --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
        --tokenizer-type BertWordPieceCase \
        --vocab-file $VOCAB_FILE \
        --num-layers 48 \
        --hidden-size 2560 \
        --num-attention-heads 40 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH/merged