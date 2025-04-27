#!/bin/bash
#SBATCH -A trn040
#SBATCH -J llm_inference
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
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

# Set OMP and MKL threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Choose one of the following execution modes by uncommenting:

# 1. Run in interactive mode (good for testing):
# python inference.py --gpu --bf16 --interactive

# 2. Process a single question:
# python inference.py --gpu --bf16 --question "Write a function to calculate the factorial of a number in Python."

# 3. Batch process from file (one question per line in input.txt, one answer per line in output.txt):

$CONDA_ENV/bin/python inference.py \
    --gpu \
    --bf16 \
    --input-file questions.txt \
    --output-file answers.txt 