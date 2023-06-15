#!/bin/bash
#SBATCH --job-name=bert8gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition gputest
#SBATCH -t 00:15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4
#SBATCH --ntasks-per-node=1
#SBATCH --distribution=block:block
#SBATCH --account=project_2004600
#SBATCH -o logs/bert-%x-%j.out
#SBATCH -e logs/bert-%x-%j.err

set -euo pipefail

unset samples
unset flops

rm -f logs/latest.out logs/latest.err
ln -s bert-$SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s bert-$SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err


module load pytorch
module load openmpi

#OMP_THREADS/CPU
OMP_DISPLAY_AFFINITY=true
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=verbose

#CUDA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_LAUNCH_BLOCKING=1
echo "CUDA DEVICES FROM SBATCH" $CUDA_VISIBLE_DEVICES

#Uncomment for debugging
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL


#TODO VAIHDA MASTER_ADDR hostname head -n 1 tms
#Distributed args
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))



#echo "LOCAL ID FROM SBATCH" $SLURM_LOCALID
echo "START $SLURM_JOBID: $(date)"



export TYPE='bert'
export USER=villekom
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/
#TP_SIZE=1
#PP_SIZE=1
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))


#MODEL_ARGS
BATCH_SIZE_PER_GPU=30
LR=1e-4
MIN_LR=1e-5
export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*BATCH_SIZE_PER_GPU))



export TENSORBOARD_DIR="logs/$TYPE/tb_logs/ngpus_$WORLD_SIZE"
rm -rf $TENSORBOARD_DIR

CHECKPOINT_PATH=checkpoints/bert_345m
rm -rf $CHECKPOINT_PATH
VOCAB_FILE=/scratch/project_2004600/FinBERT-data/finbert-cased-preprocessed/finbert-base-tokenizer/vocab.txt
DATA_PATH=data/bert/finbert_data/train/finbert_train
BERT_ARGS="--num-layers 24 \
          --no-pipeline-parallel \
          --tensor-model-parallel-size 1 \
          --pipeline-model-parallel-size 1 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 512 \
          --split 949,50,1 \
          --max-position-embeddings 512\
          --micro-batch-size $BATCH_SIZE_PER_GPU \
          --global-batch-size $GLOBAL_BATCH_SIZE \
          --lr 1.5e-4 \
          --train-iters 5000 \
          --save $CHECKPOINT_PATH \
          --load $CHECKPOINT_PATH
          --vocab-file $VOCAB_FILE \
          --tokenizer-type BertWordPieceCase \
          --data-impl mmap \
          --num-workers $SLURM_CPUS_PER_TASK \
          --data-path $DATA_PATH \
          --DDP-impl torch \
          --lr-warmup-fraction .01 \
          --fp16 \
          --world_size $WORLD_SIZE \
          "

OUTPUT_ARGS="--log-interval 100 \
             --save-interval 100 \
             --eval-interval 200 \
             --eval-iters 10 \
             --tensorboard-dir $TENSORBOARD_DIR \
             --log-batch-size-to-tensorboard \
             --log-timers-to-tensorboard \
             --tensorboard-queue-size 1 \
             "


ZERO_STAGE=1

config_json="ds_configs/./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOF > "$config_json"
{
    "train_micro_batch_size_per_gpu": $BATCH_SIZE_PER_GPU,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "fp16": {
        "enabled": true
    },
    "steps_per_print": 100,
    "wall_clock_breakdown": true,
    "flops_profiler": {
        "enabled": false
    },
     "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": true,
        "profile": false
    },
    "tensorboard": {
        "enabled": true,
        "output_path": "$TENSORBOARD_DIR",
        "job_name": "ds_logs"
    }
}
EOF


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    --synchronize-each-layer
    "
# if you can't stand pt-1.9 launcher noise
export LOGLEVEL=INFO
#export TORCH_CPP_LOG_LEVEL=INFO

export TORCH_LAUNCHER="python -m torch.distributed.run \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_backend=c10d \
    "

export DS_LAUNCHER="deepspeed --autotuning=[run|tune] --num_nodes $SLURM_NNODES --num_gpus $WORLD_SIZE"

export CMD=" \
    pretrain_bert.py \
    $BERT_ARGS \
    $OUTPUT_ARGS \
    $DEEPSPEED_ARGS \
    "

srun -l python3 $CMD

#Take out the last printed average samples_per second as it is logged times*amount of gpus(I think)
samples=$(grep "Average samples per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
#First value is the average samples from the runs followed by the amount of nodes. Seperated by whitespace
echo $samples $SLURM_NNODES >> samples.txt
flops=$(grep "Average tflops per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
echo $flops $SLURM_NNODES >> samples.txt

echo "END $SLURM_JOBID: $(date)"