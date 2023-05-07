#!/bin/bash
#SBATCH --job-name=preprocess_bert
#SBATCH --account=project_462000119
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH --partition=small

source /scratch/project_462000185/ville/megatron_repos/Ville_Megatron-DeepSpeed/venv/bin/activate


BS_DIR=/pfs/lustrep4/scratch/project_462000119/ville/BS_Megatron/tools
DATA_DIR=/pfs/lustrep4/scratch/project_462000119/data/FinBERT/jsonl/train/*



srun python3 $BS_DIR/preprocess_data_many_cores.py \
       --input /scratch/project_462000185/ville/data/Finbert_data/combined/balanced_weighted_jsonl/merged.jsonl \
       --output-prefix /scratch/project_462000185/ville/data/Finbert_data/combined/converted/old_tokenizer/single_weighted/balanced_finbert_data \
       --vocab bert/finbert_vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceCase \
       --split-sentences \
       --workers $SLURM_CPUS_PER_TASK