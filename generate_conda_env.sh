#!/bin/bash

ROCM_VERSION="6.3.1"
PROJECT_DIR="/gpfs/wolf2/olcf/trn040/world-shared"
CONDA_ENV=${PROJECT_DIR}/pytorch-rocm-${ROCM_VERSION}

# Load modules
module load PrgEnv-gnu/8.6.0
module load miniforge3
module load rocm/${ROCM_VERSION}
module load craype-accel-amd-gfx90a

if [ -d "${CONDA_ENV}" ]; then
    echo "Conda environment ${CONDA_ENV} already exists"
    echo "Activating the environment"
    source activate ${CONDA_ENV}
    exit 0
fi

# Create a new conda environment
conda create -p ${CONDA_ENV} python=3.11

# Activate the environment
source activate ${CONDA_ENV}

# Install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION%.*}

# Install other dependencies
pip install datasets
pip install transformers
pip install peft
pip install accelerate
pip install bitsandbytes
pip install xformers
pip install -U "huggingface_hub[cli]"