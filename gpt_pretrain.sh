#!/bin/bash
#SBATCH --job-name=meg_gpt_8gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=pilot
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --distribution=block:block
#SBATCH --account=project_462000119
#SBATCH --threads-per-core=2
#SBATCH -o logs/gpt-%x.out
#SBATCH -e logs/gpt-%x.err


set -euo pipefail

unset samples
unset flops

#mkdir -p logs
rm -f logs/latest.out logs/latest.err
ln -s gpt-$SLURM_JOB_NAME.out logs/latest.out
ln -s gpt-$SLURM_JOB_NAME.err logs/latest.err

module purge
module load PrgEnv-gnu
module load cray-python
module load cray-mpich
module load rocm
module load craype-x86-trento
module load craype-accel-amd-gfx90a



source /projappl/project_462000119/ville/Ville_Megatron-DeepSpeed/venv/bin/activate

set | grep SLURM | while read line; do echo "# $line"; done

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
#TORCH-EXTENSIONS
export USER=villekom
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/
CHECKPOINT_PATH=/scratch/project_462000119/ville/checkpoints
rm -rf $CHECKPOINT_PATH

#BINDING/OPTIMIZATION AFFINITY
export MPICH_OFI_NIC_POLICY=GPU

#OMP_THREADS/CPU
OMP_DISPLAY_AFFINITY=true
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
#export OMP_NUM_THREADS=2
export SLURM_CPU_BIND=verbose
#export SLURM_GPU_BIND=verbose

#CUDA AND TORCH_DDP
export CUDA_DEVICE_ORDER=PCI_BUS_ID


#DEBUG
#export NCCL_DEBUG=INFO
#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED=DETAIL
#export CUDA_LAUNCH_BLOCKING=1

#WORLD_SIZE for deepspeed/torch distributed init
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))


#export RANK=$SLURM_PROCID
#export LOCAL_RANK=$SLURM_LOCALID
echo "ROCR_VISIBLE_DEVICES"=$ROCR_VISIBLE_DEVICES
echo "WORLD_SIZE" $WORLD_SIZE
echo "SLURM_NODEID" $SLURM_NODEID
echo "SLURM LOCALID" $SLURM_LOCALID
echo "SLURM PROCID" $SLURM_PROCID
echo "SLURM_JOB_GPUS" $SLURM_JOB_GPUS

#NETWORK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=1234


# Model
export TYPE='gpt2'
SEQ_LENGTH=1024
export TENSORBOARD_DIR="logs/$TYPE/tb_logs/ngpus_$WORLD_SIZE"
# Model-specific args

# gpt2-small: 12 layers, hidden size 768, 12 attention heads
# NUM_LAYERS=12
# HIDDEN_SIZE=768
# NUM_ATTENTION_HEADS=12
# MICRO_BATCH_SIZE_PER_GPU=16

# gpt2-medium: 24 layers, hidden size 1024, 16 attention heads
export NUM_LAYERS=24
export HIDDEN_SIZE=1024
export NUM_ATTENTION_HEADS=16
export MICRO_BATCH_SIZE_PER_GPU=8

# gpt2-large: 36 layers, hidden size 1280, 20 attention heads
# NUM_LAYERS=36
# HIDDEN_SIZE=1280
# NUM_ATTENTION_HEADS=20
# MICRO_BATCH_SIZE_PER_GPU=4

# Data
VOCAB_FILE=gpt2/gpt2-vocab.json
MERGE_FILE=gpt2/gpt2-merges.txt
DATA_PATH=data/gpt2/bookcorpus/BookCorpusDataset_text_document

# Training
TRAIN_ITERS=1000
LEARNING_RATE=0.00015
ZERO_STAGE=1
GRADIENT_ACCUMULATION_STEPS=1
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1

mkdir -p ds_configs
config_json="ds_configs/./ds_config.$SLURM_JOBID.json"
#TODO
#Autotune
cat <<EOF > "$config_json"
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE_PER_GPU,
    "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "fp16": {
        "enabled": true
    },
    "steps_per_print": 50000,
    "wall_clock_breakdown": true,
    "flops_profiler": {
        "enabled": false
    },
     "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    "tensorboard": {
        "enabled": true,
        "output_path": "$TENSORBOARD_DIR/ds_logs",
        "job_name": "ngpus_$WORLD_SIZE"
    },
    "elasticity": {
    "enabled": true,
    "max_train_batch_size": $SEQ_LENGTH,
    "micro_batch_sizes": [2,4,8],
    "min_gpus": $WORLD_SIZE,
    "max_gpus": $WORLD_SIZE,
    "min_time": 0,
    "version": 0.2,
    "ignore_non_elastic_batch_info": true,
    "num_gpus_per_node": 8,
    "model_parallel_size": $TENSOR_PARALLEL
  }
}

EOF

GPT_ARGS="--num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTENTION_HEADS \
        --max-position-embeddings $SEQ_LENGTH \
        --encoder-seq-length $SEQ_LENGTH \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --micro-batch-size $MICRO_BATCH_SIZE_PER_GPU \
        --tensor-model-parallel-size $TENSOR_PARALLEL \
        --pipeline-model-parallel-size $PIPELINE_PARALLEL \
        --train-iters 5000 \
        --lr $LEARNING_RATE \
        --lr-warmup-fraction 0.01 \
        --num-workers $SLURM_CPUS_PER_TASK \
        --tokenizer-type GPT2BPETokenizer \
        --data-path $DATA_PATH \
        --world_size $WORLD_SIZE \
        --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --tensorboard-dir $TENSORBOARD_DIR \
             --log-batch-size-to-tensorboard \
             --log-timers-to-tensorboard \
             "

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export TORCH_LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --nnodes $SLURM_NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export DS_LAUNCHER="deepspeed --num_nodes $SLURM_NNODES --num_gpus $WORLD_SIZE"

export CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $DEEPSPEED_ARGS \
    "


echo "START $SLURM_JOBID: $(date)"

MASKS="ff000000000000,ff00000000000000,ff0000,ff000000,ff,ff00,ff00000000,ff0000000000"

srun -l --cpu-bind=mask_cpu:$MASKS python3 $CMD

#Take out the last printed average samples_per second as it is logged times*amount of gpus(I think)
samples=$(grep "Average samples per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
#First value is the average samples from the runs followed by the amount of nodes. Seperated by whitespace
echo $samples $SLURM_NNODES >> samples.txt
flops=$(grep "Average tflops per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
echo $flops $SLURM_NNODES >> samples.txt



echo "END $SLURM_JOBID: $(date)"