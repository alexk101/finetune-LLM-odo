import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned LLaMA model")
    parser.add_argument("--base-model", type=str, default="meta-llama/Meta-Llama-3-8B", 
                        help="Base model name")
    parser.add_argument("--adapter-path", type=str, default="finetuned_llama", 
                        help="Path to the fine-tuned LoRA adapter")
    parser.add_argument("--max-length", type=int, default=4096, 
                        help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    parser.add_argument("--question", type=str, default=None, 
                        help="Question to process (for non-interactive mode)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Input file with questions (one per line)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output file for responses (one per line)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    return parser.parse_args()

def format_prompt(question):
    """Format the prompt similar to the training data format"""
    return f"Question: {question}\n\nSolution:"

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load the base model
    print("Loading base model...")
    model_kwargs = {"device_map": "auto"} if args.gpu else {}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # Load the fine-tuned LoRA adapter
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    # Inference function
    def generate_answer(question):
        formatted_prompt = format_prompt(question)
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=1,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the solution part (remove the original prompt)
        solution = generated_text[len(formatted_prompt):]
        return solution.strip()
    
    # Run batch inference from file
    if args.input_file and args.output_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} does not exist.")
            return
        
        print(f"Processing questions from {args.input_file}...")
        questions = []
        with open(args.input_file, 'r') as infile:
            questions = [line.strip() for line in infile if line.strip()]
        
        print(f"Found {len(questions)} questions to process.")
        
        # Process questions and save responses
        with open(args.output_file, 'w') as outfile:
            for i, question in enumerate(tqdm(questions)):
                try:
                    solution = generate_answer(question)
                    outfile.write(f"{solution}\n")
                    outfile.flush()  # Ensure the output is written immediately
                except Exception as e:
                    print(f"Error processing question {i+1}: {e}")
                    outfile.write(f"ERROR: {str(e)}\n")
                    outfile.flush()
        
        print(f"Responses saved to {args.output_file}")
    
    # Run in interactive mode
    elif args.interactive:
        print("=" * 50)
        print("Interactive mode. Type 'exit' to quit.")
        print("=" * 50)
        
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == "exit":
                break
            
            if not question.strip():
                continue
            
            print("\n" + format_prompt(question))    
            solution = generate_answer(question)
            print("\nSolution:")
            print("-" * 50)
            print(solution)
            print("-" * 50)
    
    # Process a single question
    elif args.question:
        print("\n" + format_prompt(args.question))
        solution = generate_answer(args.question)
        print("\nSolution:")
        print("-" * 50)
        print(solution)
        print("-" * 50)
    
    else:
        print("Please provide one of the following:")
        print("  - Input and output files (--input-file and --output-file)")
        print("  - A single question (--question)")
        print("  - Interactive mode (--interactive)")

if __name__ == "__main__":
    main() 