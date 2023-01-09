#!/bin/bash

python3 zero_to_fp32_ville.py --checkpoint_dir <"path to the desired checkpoint folder, e.g., path/checkpoint-12"> \
                                --output_file <"path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)">
