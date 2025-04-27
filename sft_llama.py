import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from comm import init_process_group
import torch.distributed as dist


# Initialize distributed training
rank, world_size, local_rank = init_process_group()

# Set device for this process - always use cuda:0 since that's all each process sees
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Model and tokenizer names
base_model_name = "meta-llama/Meta-Llama-3-8B"
new_model_name = "llama-3-8b-enhanced" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(
    base_model_name, 
    trust_remote_code=True,
    use_fast=False
)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Model - remove explicit device mapping for distributed training
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    cache_dir="llama3-models"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Use streaming mode to process data on-the-fly instead of preprocessing everything at once
data_name = "nvidia/OpenCodeReasoning"

# We'll use streaming mode with a data formatting function
# This processes examples on-the-fly instead of transforming the entire dataset at once
def formatting_func(examples):
    """Format the examples into the desired SFT format"""
    texts = []
    for i in range(len(examples["input"])):
        # Format each example with input and solution
        text = f"Question: {examples['input'][i]}\n\nSolution: {examples['solution'][i]}"
        texts.append(text)
    return {"text": texts}

# Load the streaming dataset
streaming_dataset_dict = load_dataset(
    data_name, 
    "split_0", 
    streaming=True,
    cache_dir="data_cache"
)

# Extract the specific split from the IterableDatasetDict
streaming_dataset = streaming_dataset_dict["split_0"]

# Set up streaming with batched processing
streaming_dataset = streaming_dataset.map(
    formatting_func,
    batched=True,
    batch_size=10  # Process 10 examples at a time
)

# Convert to regular dataset for SFTTrainer but with a reasonable subset size
# This allows us to process in chunks rather than all at once
max_samples = 50000  # Adjust based on memory constraints
training_data = streaming_dataset.take(max_samples)

# Only print dataset info from main process
if int(os.environ.get("LOCAL_RANK", "0")) == 0:
    # Get one sample to show
    sample = next(iter(streaming_dataset))
    print("Sample formatted example:")
    print(sample["text"][:500] + "...")
    
    # Show memory usage before training
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

# Training Params with gradient checkpointing to reduce memory usage
train_params = SFTConfig(
    output_dir="./results_modified",
    num_train_epochs=200,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=20000,  # Limit number of steps instead of epochs for large datasets
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
    # Add distributed training parameters
    ddp_find_unused_parameters=False,
    remove_unused_columns=True,
    # Add label_names to address warning
    label_names=[],
    # Enable gradient checkpointing to save memory
    gradient_checkpointing=True
)

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    processing_class=llama_tokenizer,
    args=train_params
)

print("Training...")
# Training
fine_tuning.train()

# Save the model in the main process only
if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
        fine_tuning.model.save_pretrained("finetuned_llama")
else:
    fine_tuning.model.save_pretrained("finetuned_llama")
