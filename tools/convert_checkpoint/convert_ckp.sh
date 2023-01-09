#!/bin/bash

module --quiet purge
module load cray-python
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
python -m pip install transformers datasets scikit-learn

#Give the path containing the mp_rank_00_model_states.pt file as the first parameter
#Only tested with tp=1

python3 convert_megatron_bert_checkpoint.py path/to/mp_rank_00_model_states.pt
                                            #Make sure its the latest iteration checkpoint directory
