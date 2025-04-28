# LLaMA-3 Fine-tuning and Inference for Code Reasoning

This repository contains code for fine-tuning LLaMA-3 models on code reasoning tasks and running inference with the fine-tuned models. It is optimized for running on Frontier at ORNL. Few parts of this code have been adapted from this (repo)[https://github.com/0traced/frontier-finetuning]. Fine tuning on 20000 samples takes ~3-4 hours on one gpu.

**NOTE**

Multinode+multigpu training is not yet supported. Please use only single GPU implementations.

## Dataset

The model is finetuned on Nvidia's (OpenCodeReasoning)[https://huggingface.co/datasets/nvidia/OpenCodeReasoning] dataset, which is a collection of questions from multiple coding datasets, and answers from nvidia's Open-R1 model. You can try fine tuning on the `output` column, which is the output of the chain of thought model, or the `solution` column, which is just the original solution to the problem. You can see examples of each in `example/subset.csv`.

If you use the `output` column, since llama-3 (which is what we were asked to use) is not a chain-of-thought reasoning model, this dataset won't necessarily improve the model in any way. In fact, it provides a very good example of why it is important to match a model's actual structure with its training data. In this case, if you fine tune a regular LLM on output from a reasoning model, what you end up with is typically a lot of repetition. You can see this in the example answers. Though, the base model also seems to be somewhat bad at some of the questions I have setup, especially when I ask it to implement something in D. If you fine tune on the `solution` column, you see less of a difference in the output, since it matches what a traditional model expects, that is question + answer.

## Repository Structure

- **`sft_llama.py`**: Main script for fine-tuning LLaMA-3 models with LoRA adapters
- **`comm.py`**: Utilities for distributed training communication (for future use)
- **`inference.py`**: Script for running inference with fine-tuned models
- **`submit.sh`**: SLURM job submission script for fine-tuning 
- **`submit_single_gpu.sh`**: SLURM job submission script for fine-tuning on one GPU (recommended)
- **`run_inference.sh`**: SLURM job submission script for inference
- **`export_DDP_vars.sh`**: Environment variable setup for distributed training (for future use)
- **`questions.txt`**: Example questions for testing inference
- **`example`**: This directory contains examples of fine-tuning outputs (on both `output` and `solution` columns), inference results comparing base and fine-tuned models, and sample dataset files

## Prerequisites

- Access to Frontier at ORNL
- Python environment with PyTorch, Transformers, and PEFT installed
- Access to the LLaMA-3 model weights
- Hugging Face account with model access permissions

## Fine-tuning

The `sft_llama.py` script fine-tunes Meta's LLaMA-3-8B model on the OpenCodeReasoning dataset using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

### Key Features

- ~~Distributed training support via PyTorch DDP~~ (Coming in future updates)
- Memory-efficient streaming dataset processing
- LoRA fine-tuning for parameter efficiency
- Support for AMD GPUs on Frontier

### Running Fine-tuning

To submit a single-GPU fine-tuning job (recommended):

```bash
sbatch submit_single_gpu.sh
```

The script will:
1. Load and stream the OpenCodeReasoning dataset
2. Set up LoRA adapters for efficient fine-tuning
3. Train the model with the specified hyperparameters
4. Save the fine-tuned model to "finetuned_llama" directory

### Configuration

Edit `sft_llama.py` to customize:

- Model: Base model name and configuration
- Training: Batch size, learning rate, number of epochs
- LoRA: Rank, alpha, and dropout values

## Inference

The `inference.py` script allows you to run inference with the fine-tuned model.

### Key Features

- Interactive mode for single queries
- Batch processing from file with one question per line
- GPU acceleration with bfloat16 precision support
- Error handling for robust batch processing

### Running Inference

There are three ways to run inference:

1. **Interactive Mode**:
   ```bash
   python inference.py --gpu --bf16 --interactive
   ```

2. **Single Question**:
   ```bash
   python inference.py --gpu --bf16 --question "Write a function to calculate the factorial of a number in Python."
   ```

3. **Batch Processing**:
   ```bash
   python inference.py --gpu --bf16 --input-file questions.txt --output-file answers.txt
   ```

To submit as a job:
```bash
sbatch run_inference.sh
```

### Command-line Arguments

- `--base-model`: Base model name (default: "meta-llama/Meta-Llama-3-8B")
- `--adapter-path`: Path to fine-tuned adapter (default: "finetuned_llama")
- `--max-length`: Maximum output length (default: 4096)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--input-file`: Input file with questions (one per line)
- `--output-file`: Output file for responses
- `--gpu`: Use GPU for inference
- `--bf16`: Use bfloat16 precision

## Future Distributed Training Support

> **Note**: Multi-GPU and multi-node training is currently under development and not yet fully supported.

The repository includes initial setup for PyTorch's Distributed Data Parallel (DDP):

- `comm.py` handles the initialization of the process group
- `export_DDP_vars.sh` sets up the necessary environment variables
- AMD GPU visibility is controlled via ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES

Currently, it is recommended to use the single GPU implementation via `submit_single_gpu.sh`.

## Example Questions

The repository includes a `questions.txt` file with example programming questions across various languages and concepts. You can use this as a template to create your own question files.

## Frontier-Specific Configuration

The scripts are optimized for Frontier's AMD GPUs with:

- Proper module loading in job scripts
- Environment variable setup for AMD GPU visibility
- BFloat16 precision support for AMD MI250X GPUs
- Proxy settings for network access

## Output

- Fine-tuning: Saves models to the "finetuned_llama" directory
- Inference: Outputs predictions to console or to a file (one answer per line)

## Troubleshooting

For memory issues:
- Reduce batch size
- Enable gradient checkpointing
- Use a smaller subset of the dataset for initial testing
