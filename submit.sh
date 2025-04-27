#!/bin/bash
#SBATCH -A trn040
#SBATCH -J llm
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8  # 8 tasks per node for 8 MI250X GPUs
#SBATCH --cpus-per-task=7
#SBATCH --signal=B:USR1@60  # Send signal 10 minutes before time limit
#SBATCH -o %x-%j.out

module load PrgEnv-gnu/8.6.0
module load miniforge3
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

CONDA_ENV="/ccsopen/home/pwk/scratch/pytorch-rocm-6.3.1"
source activate ${CONDA_ENV}

export HF_DATASETS_CACHE="$HOME/scratch/.huggingface"

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=3442

set +x

CMD="accelerate launch \
    --num_processes $((SLURM_NNODES * 8)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    sft_llama.py \
    "
srun $CMD
# srun ${CONDA_ENV}/bin/python sft_llama.py 
