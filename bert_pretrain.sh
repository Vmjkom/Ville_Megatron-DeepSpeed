#!/bin/bash
#SBATCH --job-name=BertXL_32gpu_bigrun
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=pilot
#SBATCH -t 2-00:00:00
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --distribution=block:block
#SBATCH --account=project_462000119
#SBATCH --threads-per-core=2
#SBATCH -o logs/bert-%x-%j.out
#SBATCH -e logs/bert-%x-%j.err

set -euo pipefail

unset samples
unset flops

rm -f logs/latest.out logs/latest.err
ln -s bert-$SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s bert-$SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err


module purge
module load cray-python
module load PrgEnv-amd
module load cray-mpich
module load craype-x86-trento
module load craype-accel-amd-gfx90a

source /scratch/project_462000119/ville/Ville_Megatron-DeepSpeed/venv/bin/activate

set | grep SLURM | while read line; do echo "# $line"; done

#FAULTY SOCKETS
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3


#BINDING/OPTIMIZATION AFFINITY
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1 

#OMP_THREADS/CPU
#OMP_DISPLAY_AFFINITY=true
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=verbose

#CUDA/rocr
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_LAUNCH_BLOCKING=1
unset ROCR_VISIBLE_DEVICES

#Uncomment for debugging
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL


#Distributed args
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

DATE=$(date "+%Y.%m.%d-%H.%M.%S")
echo "START $SLURM_JOBID: $DATE"


export TYPE='bert'
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/
#rm -rf $TORCH_EXTENSIONS_DIR
#TP_SIZE=1
#PP_SIZE=1

#model_config
#BERT 110M (same config as original BERT-Base model)
## This config is not included in Megatron-LM paper
#model_size=110M
#num_layers=12
#hidden_size=768
#num_attn_heads=12
#init_std=0.02

## BERT 345M (same config as original BERT-Large model)
#model_size=345M
#num_layers=24
#hidden_size=1024
#num_attn_heads=16
#init_std=0.02

## BERT 1.3B
model_size=1.3B
num_layers=24
hidden_size=2048
num_attn_heads=32
init_std=0.013

## BERT 3.9B
# model_size=3.9
# num_layers=48
# hidden_size=2560
# num_attn_heads=40
# init_std=0.011

export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))
export BATCH_SIZE_PER_GPU=64
export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*BATCH_SIZE_PER_GPU))

export TENSORBOARD_DIR="logs/tb_logs/$TYPE/$model_size/ngpus_${WORLD_SIZE}/$SLURM_JOB_NAME"



CHECKPOINT_PATH=checkpoints/$TYPE/$model_size/$SLURM_JOB_NAME
#rm -rf $CHECKPOINT_PATH
VOCAB_FILE=/scratch/project_462000119/ville/Ville_Megatron-DeepSpeed/bert/finbert_vocab.txt
DATA_PATH=/scratch/project_462000119/ville/Ville_Megatron-DeepSpeed/data/bert/finbert_data/finbert_train

BERT_ARGS="--adam-beta1 0.9 \
            --adam-beta2 0.999 \
            --num-layers $num_layers \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 1 \
            --no-pipeline-parallel \
            --hidden-size $hidden_size \
            --init-method-std ${init_std} \
            --num-attention-heads $num_attn_heads \
            --seq-length 512 \
            --split 949,50,1 \
            --max-position-embeddings 512\
            --micro-batch-size $BATCH_SIZE_PER_GPU \
            --global-batch-size $GLOBAL_BATCH_SIZE \
            --lr 1e-4 \
            --min-lr 1e-5 \
            --weight-decay 1e-2 \
            --checkpoint-activations \
            --lr-decay-style linear \
            --train-iters 1000000 \
            --vocab-file $VOCAB_FILE \
            --tokenizer-type BertWordPieceCase \
            --data-impl mmap \
            --num-workers 4 \
            --data-path $DATA_PATH \
            --DDP-impl torch \
            --lr-warmup-fraction .01 \
            --fp16 \
            --world_size $WORLD_SIZE \
            --save $CHECKPOINT_PATH \
          "
#TODO Layer norm epsilon change
OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10 \
             --tensorboard-dir $TENSORBOARD_DIR \
             --log-batch-size-to-tensorboard \
             --log-timers-to-tensorboard \
             "


ZERO_STAGE=0

config_json="ds_configs/./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOF > "$config_json"
{
    "train_micro_batch_size_per_gpu": $BATCH_SIZE_PER_GPU,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "elastic_checkpoint": true
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 500,
        "hysteresis": 2,
        "min_loss_scale": 1,
        "initial_scale_power": 11
    },
    "steps_per_print": 100,
    "wall_clock_breakdown": true,
    "flops_profiler": {
        "enabled": false
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
    "
# if you can't stand pt-1.9 launcher noise
export LOGLEVEL=INFO
#export TORCH_CPP_LOG_LEVEL=INFO

export TORCH_LAUNCHER="python -m torch.distributed.run \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_backend=c10d \
    "

export DS_LAUNCHER="deepspeed --num_nodes $SLURM_NNODES --num_gpus $WORLD_SIZE" # Does not work with lumi

export CMD=" \
    pretrain_bert.py \
    $BERT_ARGS \
    $OUTPUT_ARGS \
    $DEEPSPEED_ARGS \
    "

c=fe
MASKS=0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000
#MASKS="ff000000000000,ff00000000000000,ff0000,ff000000,ff,ff00,ff00000000,ff0000000000"


srun -l --cpu-bind=mask_cpu:$MASKS python3 $CMD

#Take out the last printed average samples_per second as it is logged times*amount of gpus(I think)
samples=$(grep "Average samples per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
#First value is the average samples from the runs followed by the amount of nodes. Seperated by whitespace
echo $samples $SLURM_NNODES >> samples.txt
flops=$(grep "Average tflops per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
echo $flops $SLURM_NNODES >> samples.txt

echo "END $SLURM_JOBID: $(date)"