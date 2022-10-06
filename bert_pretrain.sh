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
#SBATCH -o logs/bert-%x.out
#SBATCH -e logs/bert-%x.err

set -euxo pipefail

unset samples
unset flops

rm -f logs/latest.out logs/latest.err
ln -s bert-$SLURM_JOB_NAME.out logs/latest.out
ln -s bert-$SLURM_JOB_NAME.err logs/latest.err

module purge


module load pytorch/1.12
#module load pgi/19.7
#module load cuda
#module load hpcx-mpi

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

#Uncomment for debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL


#TODO VAIHDA MASTER_ADDR hostname head -n 1 tms
#Distributed args
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))



#echo "SLURM GPUS OVERALL SBATCH" $SLURM_JOB_GPUS

#echo "LOCAL ID FROM SBATCH" $SLURM_LOCALID
echo "START $SLURM_JOBID: $(date)"


#set | grep MPI | while read line; do echo "# $line"; done
#set | grep SLURM | while read line; do echo "# $line"; done

export TYPE='bert'
export USER=villekom
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/
#TP_SIZE=1
#PP_SIZE=1

export NNODES=$SLURM_NNODES
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

#CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
#echo "SLURM GPUS STEP VS JOB" $SLURM_STEP_GPUS $SLURM_JOB_GPUS
#echo "SLURM DEVICE ORDINAL " $GPU_DEVICE_ORDINAL
#echo "SLURM GPUS OVERALL BERT" $SLURM_GPUS
#CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ORDINAL


BATCH_SIZE_PER_GPU=15
#export RANK=$SLURM_PROCID
#export LOCAL_RANK=$SLURM_LOCALID
#echo "CUDA_VISIBLE_DEVICES"=$CUDA_VISIBLE_DEVICES
#echo "WORLD_SIZE" $WORLD_SIZE
#echo "SLURM_NODEID" $SLURM_NODEID
#echo "SLURM LOCALID" $SLURM_LOCALID
#echo "SLURM PROCID" $SLURM_PROCID
#echo "MPIX_RANK" $MPIX_RANK




export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*BATCH_SIZE_PER_GPU))

export TENSORBOARD_DIR="logs/$TYPE/tb_logs/ngpus_$WORLD_SIZE"
rm -rf $TENSORBOARD_DIR
#OMP_NUM_THREADS=2


CHECKPOINT_PATH=checkpoints/bert_345m
rm -rf $CHECKPOINT_PATH
VOCAB_FILE=nvidia/megatron-bert-cased-345m/vocab.txt
DATA_PATH=data/bert/tiny-owt2-sample_text_sentence
BERT_ARGS="--num-layers 24 \
          --no-pipeline-parallel \
          --pipeline-model-parallel-size 1 \
          --tensor-model-parallel-size 1 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 512 \
          --split 949,50,1 \
          --max-position-embeddings 512\
          --micro-batch-size $BATCH_SIZE_PER_GPU \
          --global-batch-size $GLOBAL_BATCH_SIZE \
          --lr 0.0001 \
          --train-iters 100 \
          --lr-decay-iters 99 \
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

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
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
    --nnodes $NNODES \
    --rdzv_backend=c10d \
    "

export DS_LAUNCHER="deepspeed --num_nodes $NNODES --num_gpus $WORLD_SIZE"

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