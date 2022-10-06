#!/bin/bash
#SBATCH --job-name=meg_gpt_8gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --partition gputest
#SBATCH -t 00:15:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=v100:4
##SBATCH --gpu-bind=single:1
#SBATCH --ntasks-per-node=4
#SBATCH --distribution=block:block
#SBATCH --account=project_2004600
#SBATCH -o logs/gpt-%x.out
#SBATCH -e logs/gpt-%x.err


set -euxo pipefail

unset samples
unset flops

rm -f logs/latest.out logs/latest.err
ln -s bert-$SLURM_JOB_NAME.out logs/latest.out
ln -s bert-$SLURM_JOB_NAME.err logs/latest.err

module purge

module load pytorch/1.12


set | grep SLURM | while read line; do echo "# $line"; done


#TORCH-EXTENSIONS
export USER=villekom
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/


#OMP_THREADS/CPU
#OMP_DISPLAY_AFFINITY=true
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SLURM_CPU_BIND=verbose

#CUDA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_LAUNCH_BLOCKING=1
echo "CUDA DEVICES FROM SBATCH" $CUDA_VISIBLE_DEVICES

#WORLD_SIZE for deepspeed/torch distributed init
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))
export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*BATCH_SIZE_PER_GPU))

#export BATCH_SIZE_PER_GPU=15
#export RANK=$SLURM_PROCID
#export LOCAL_RANK=$SLURM_LOCALID
echo "CUDA_VISIBLE_DEVICES"=$CUDA_VISIBLE_DEVICES
echo "WORLD_SIZE" $WORLD_SIZE
echo "SLURM_NODEID" $SLURM_NODEID
echo "SLURM LOCALID" $SLURM_LOCALID
echo "SLURM PROCID" $SLURM_PROCID
echo "SLURM_JOB_GPUS" $SLURM_JOB_GPUS


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
VOCAB_FILE=$TYPE/gpt2-vocab.json
MERGE_FILE=$TYPE/gpt2-merges.txt
DATA_PATH=data/$TYPE/tiny-owt-sample_text_sentence_text_document

# Training
TRAIN_ITERS=100
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
    "train_micro_batch_size_per_gpu": $BATCH_SIZE_PER_GPU,
    "gradient_clipping": 1.0,
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
    "tensorboard": {
        "enabled": false,
        "output_path": "logs/tb_logs/ds_logs/",
        "job_name": "$SLURM_GPUS_ON_NODE"
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
        --micro-batch-size $MICRO_BATCH_SIZE_PER_GPU \
        --tensor-model-parallel-size $TENSOR_PARALLEL \
        --pipeline-model-parallel-size $PIPELINE_PARALLEL \
        --train-iters $TRAIN_ITERS \
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
    "

export TORCH_LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export DS_LAUNCHER="deepspeed --num_nodes $NNODES --num_gpus $WORLD_SIZE"

export CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $DEEPSPEED_ARGS \
    "

echo "START $SLURM_JOBID: $(date)"

#TODO 
#cpu-bind to gpus
srun -l python3 $CMD

#Take out the last printed average samples_per second as it is logged times*amount of gpus(I think)
samples=$(grep "Average samples per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
#First value is the average samples from the runs followed by the amount of nodes. Seperated by whitespace
echo $samples $SLURM_NNODES >> samples.txt
flops=$(grep "Average tflops per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
echo $flops $SLURM_NNODES >> samples.txt

echo "END $SLURM_JOBID: $(date)"