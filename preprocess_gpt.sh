#!/bin/bash

python3 tools/preprocess_data.py \
    --input data/tiny-owt2-sample.jsonl \
    --output-prefix tiny-owt-sample_text_sentence \
    --vocab gpt2/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2/gpt2-merges.txt \
    --append-eod \
    --workers $SLURM_CPUS_PER_TASK