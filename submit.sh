#!/bin/bash
#SBATCH -A trn040
#SBATCH -J llm
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
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

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

# Set OMP and MKL threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

set +x

# Define the command to run
cmd="$CONDA_ENV/bin/python sft_llama.py"

# Run with srun using the proper format
srun -n$((SLURM_JOB_NUM_NODES*8)) \
    -c $SLURM_CPUS_PER_TASK \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --ntasks-per-node=8 \
    bash -c "
    source export_DDP_vars.sh
    $cmd
    "
