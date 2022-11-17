#!/bin/bash
#SBATCH --job-name=bert_8gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --partition=pilot
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --distribution=block:block
#SBATCH --account=project_462000119
#SBATCH --threads-per-core=2
#SBATCH -o logs/bert-%j.out
#SBATCH -e logs/bert-%j.err

set -euo pipefail

unset samples
unset flops

rm -f logs/latest.out logs/latest.err
ln -s bert-$SLURM_JOB_ID.out logs/latest.out
ln -s bert-$SLURM_JOB_ID.err logs/latest.err


module purge
module load PrgEnv-amd
module load cray-mpich
module load rocm
module load craype-x86-trento
module load craype-accel-amd-gfx90a

source venv/bin/activate

set | grep SLURM | while read line; do echo "# $line"; done

#FAULTY SOCKETS
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3


#BINDING/OPTIMIZATION AFFINITY
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1 

#OMP_THREADS/CPU
OMP_DISPLAY_AFFINITY=true
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close
#export OMP_NUM_THREADS=2
export SLURM_CPU_BIND=verbose

#CUDA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_LAUNCH_BLOCKING=1

#Uncomment for debugging
#export NCCL_DEBUG=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL


#Distributed args
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))



#echo "SLURM GPUS OVERALL SBATCH" $SLURM_JOB_GPUS

#echo "LOCAL ID FROM SBATCH" $SLURM_LOCALID
echo "START $SLURM_JOBID: $(date)"


export TYPE='bert'
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/
rm -rf $TORCH_EXTENSIONS_DIR
#TP_SIZE=1
#PP_SIZE=1


export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))
export BATCH_SIZE_PER_GPU=25
export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*BATCH_SIZE_PER_GPU))

export TENSORBOARD_DIR="logs/$TYPE/tb_logs/ngpus_$WORLD_SIZE/$SLURM_JOB_ID"
rm -rf $TENSORBOARD_DIR


CHECKPOINT_PATH=checkpoints/bert_base
rm -rf $CHECKPOINT_PATH
VOCAB_FILE=bert/finbert_vocab.txt
DATA_PATH=data/bert/finbert_data/finbert_train

BERT_ARGS="--num-layers 24 \
          --no-pipeline-parallel \
          --tensor-model-parallel-size 1 \
          --pipeline-model-parallel-size 1 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 512 \
          --split 949,5,5 \
          --max-position-embeddings 512\
          --micro-batch-size $BATCH_SIZE_PER_GPU \
          --global-batch-size $GLOBAL_BATCH_SIZE \
          --lr 1e-4 \
          --train-iters 1000000 \
          --save $CHECKPOINT_PATH \
          --vocab-file $VOCAB_FILE \
          --tokenizer-type BertWordPieceCase \
          --data-impl mmap \
          --num-workers 4 \
          --data-path $DATA_PATH \
          --DDP-impl torch \
          --lr-warmup-fraction .01 \
          --fp16 \
          --world_size $WORLD_SIZE \
          "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10 \
             --tensorboard-dir $TENSORBOARD_DIR \
             --log-batch-size-to-tensorboard \
             --log-timers-to-tensorboard \
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
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
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
    "
# if you can't stand pt-1.9 launcher noise
export LOGLEVEL=INFO
#export TORCH_CPP_LOG_LEVEL=INFO

export TORCH_LAUNCHER="python -m torch.distributed.run \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_backend=c10d \
    "

export DS_LAUNCHER="deepspeed --num_nodes $SLURM_NNODES --num_gpus $WORLD_SIZE"

export CMD=" \
    pretrain_bert.py \
    $BERT_ARGS \
    $OUTPUT_ARGS \
    $DEEPSPEED_ARGS \
    "

MASKS="ff000000000000,ff00000000000000,ff0000,ff000000,ff,ff00,ff00000000,ff0000000000"

srun -l --cpu-bind=mask_cpu:$MASKS python3 $CMD

#Take out the last printed average samples_per second as it is logged times*amount of gpus(I think)
samples=$(grep "Average samples per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
#First value is the average samples from the runs followed by the amount of nodes. Seperated by whitespace
echo $samples $SLURM_NNODES >> samples.txt
flops=$(grep "Average tflops per second" logs/latest.out | tail -n 1 | grep -o "[0-9]*\.[0-9].")
echo $flops $SLURM_NNODES >> samples.txt

echo "END $SLURM_JOBID: $(date)"