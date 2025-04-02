# CUDA_VISIBLE_DEVICES=0,1,2 trl vllm-serve \
#   --model microsoft/Phi-3-mini-4k-instruct \
#   --port 8000 \
#   --enable-prefix-caching \
#   --gpu-memory-utilization 0.90 \
#   --max-model-len 4096 \
#   --dtype half \
#   --tensor-parallel-size 3

# CUDA_VISIBLE_DEVICES=3 python train_grpo.py 

import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments
)
from trl import GRPOConfig, GRPOTrainer
from trl.core import LengthSampler
import openai
from typing import List, Dict, Any, Optional, Union
import logging
import time
from dataclasses import dataclass, field
# No need for subprocess-related imports since we're not managing the vLLM server

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GPU setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Model and training configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "grpo-phi3-qlora-output"
VLLM_PORT = 8000

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

# GRPO Configuration
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,                # Higher learning rate for QLoRA
    per_device_train_batch_size=2,     # Adjust based on GPU memory
    gradient_accumulation_steps=8,     # Effective batch size of 16 per device
    max_steps=15000,                   # ~1/4 epoch for 1M examples
    warmup_steps=1000,                 # Longer warmup for stability
    logging_steps=100,                 # More frequent logging
    save_steps=500,                    # Save checkpoints regularly
    evaluation_strategy="steps",
    eval_steps=500,
    
    # GRPO specific params
    beta=0.1,                          # KL penalty coefficient
    num_generations=4,                 # Completions per prompt for efficiency
    epsilon=0.2,                       # Clipping parameter
    scale_rewards=False,               # As recommended by Dr. GRPO paper
    
    # Generation params with vLLM
    use_vllm=True,                     # Enable vLLM for generation
    vllm_server_host="localhost",      # vLLM server host
    vllm_server_port=VLLM_PORT,        # vLLM server port
    vllm_server_timeout=120.0,         # Timeout for vLLM server connection
    temperature=0.8,                   # Generation temperature
    top_p=0.92,                        # Nucleus sampling parameter
    top_k=50,                          # Top-k filtering
    repetition_penalty=1.1,            # Reduce repetition
    max_prompt_length=384,             # Based on typical prompt length
    max_completion_length=128,         # Reasonable completion length
    
    # Optimization params
    gradient_checkpointing=True,       # Memory optimization
    fp16=True,                         # Mixed precision
    optim="adamw_torch",               # Optimizer
    weight_decay=0.01,                 # L2 regularization
    max_grad_norm=1.0,                 # Gradient clipping
    lr_scheduler_type="cosine",        # LR scheduler
    
    # Prevent overfitting
    no_cuda=False,
    seed=42,
    local_rank=-1,
    
    # Reporting & Logging
    report_to="tensorboard",
    log_level="info",
    disable_tqdm=False,
    remove_unused_columns=False,       # Important for custom datasets
    log_completions=True,
    
    # Model saving
    save_total_limit=3,                # Keep only the last 3 checkpoints
)

# GPT-4o as reward model with detailed scoring rubric
class GPT4oRewardModel:
    def __init__(self, cache_dir: Optional[str] = None):
        self.client = openai.OpenAI()
        self.system_prompt = """
        You are evaluating the quality of a response to a given system prompt and user query.
        
        The expected response format depends on how many labels are applicable:
        
        For a single label:
        "label, particular part in the system prompt which led to this label selection (explanation part for selection)"
        
        For multiple labels:
        "label1, label2, ..., labelN, explanation for label1 selection from system prompt, explanation for label2 selection from system prompt, ... and so on for each label"
        
        Rate the response on a scale from 1 to 10 based on the following criteria:
        
        1 = UNACCEPTABLE: Response is completely incorrect, potentially harmful, irrelevant to the query, or violates ethical guidelines. OR the response doesn't follow the expected format at all.
        
        2 = VERY POOR: Response is mostly incorrect, off-topic, or unhelpful. It fails to address the main points of the query and may contain significant errors. OR the format is severely incorrect (missing labels or explanations).
        
        3 = POOR: Response attempts to address the query but has major gaps, inaccuracies, or misunderstandings. The format may be partially followed but is inconsistent or unclear, especially with multiple labels.
        
        4 = BELOW AVERAGE: Response partially addresses the query but lacks depth or contains several errors. The format is attempted but not cleanly executed (e.g., unclear separation between labels and explanations).
        
        5 = AVERAGE: Response is factually adequate but basic. It addresses the main query and follows the basic format but may lack clarity in the explanations, especially with multiple labels.
        
        6 = ABOVE AVERAGE: Response is mostly accurate and provides useful information. It follows the expected format with clear labels and explanations that reference the system prompt, but may have minor organization issues with multiple labels.
        
        7 = GOOD: Response is accurate, clear, and properly formatted with well-defined labels and explanations that correctly reference relevant parts of the system prompt. All labels are accounted for.
        
        8 = VERY GOOD: Response is comprehensive, well-structured, accurate, and perfectly formatted. The labels are appropriate and each explanation clearly cites specific parts of the system prompt that justify the selection.
        
        9 = EXCELLENT: Response is exceptionally thorough and insightful while maintaining perfect format. The labels are precise and each explanation highlights the most relevant parts of the system prompt in a concise, clear manner.
        
        10 = OUTSTANDING: Response is perfect in every way - comprehensive, precise, and follows the format flawlessly. The labels are exactly right and each explanation pinpoints the exact parts of the system prompt that justify the selection most effectively.
        
        Provide only a numerical score (1-10) without any explanation.
        """
        
        # Setup caching (optional)
        self.cache_dir = cache_dir
        self.cache = {}
        if cache_dir and os.path.exists(f"{cache_dir}/reward_cache.pt"):
            logger.info("Loading reward cache from disk...")
            self.cache = torch.load(f"{cache_dir}/reward_cache.pt")
            logger.info(f"Loaded {len(self.cache)} cached rewards")
    
    def get_reward(self, system_prompt: str, user_query: str, response: str) -> float:
        # Create a cache key from the inputs
        cache_key = f"{hash(system_prompt)}-{hash(user_query)}-{hash(response)}"
        
        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"System prompt: {system_prompt}\n\nUser query: {user_query}\n\nModel response: {response}"}
            ]
            
            # Add exponential backoff for API rate limits
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=10
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"API error: {e}. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
            
            # Extract numerical score from response
            score_text = completion.choices[0].message.content.strip()
            logger.debug(f"Raw score from GPT-4o: '{score_text}'")
            logger.debug(f"For system prompt: '{system_prompt[:30]}...'")
            logger.debug(f"User query: '{user_query[:30]}...'")
            logger.debug(f"Response: '{response[:30]}...'")
            
            try:
                score = float(score_text)
                # Normalize to range [-1, 1] for RLHF
                normalized_score = (score - 5.5) / 4.5
                
                # Cache the result
                if self.cache_dir:
                    self.cache[cache_key] = normalized_score
                    if len(self.cache) % 100 == 0:  # Save cache periodically
                        os.makedirs(self.cache_dir, exist_ok=True)
                        torch.save(self.cache, f"{self.cache_dir}/reward_cache.pt")
                        
                return normalized_score
            except ValueError:
                logger.warning(f"Failed to parse score from: {score_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting reward: {e}")
            return 0.0
    
    # This method matches what GRPOTrainer expects
    def batch_evaluate(self, prompts, completions, **kwargs) -> List[float]:
        """
        Process prompts and completions as provided by GRPOTrainer and compute rewards.
        
        Args:
            prompts: List of prompts (raw data from dataset)
            completions: List of generated completions
            **kwargs: Additional arguments passed by GRPOTrainer
            
        Returns:
            List of reward scores normalized to [-1, 1]
        """
        results = []
        
        # Debug information
        if len(prompts) > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Prompt type: {type(prompts[0])}")
            logger.debug(f"Completion type: {type(completions[0])}")
            
        for prompt, completion in zip(prompts, completions):
            # Extract system prompt and user query based on the dataset format
            if isinstance(prompt, dict) and 'system_prompt' in prompt and 'user_query' in prompt:
                # Direct dictionary format
                system_prompt = prompt['system_prompt']
                user_query = prompt['user_query']
            elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
                # Conversational format
                system_prompt = next((msg["content"] for msg in prompt if msg["role"] == "system"), "")
                user_query = next((msg["content"] for msg in prompt if msg["role"] == "user"), "")
            else:
                # Plain text format - assuming system prompt and user query are separated by newlines
                # Adjust this parsing logic based on your actual data format
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                parts = prompt_str.split("\n\n")
                system_prompt = parts[0] if len(parts) > 1 else ""
                user_query = parts[-1]
            
            # Extract the completion/response
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                # Conversational format response
                response = completion[0].get("content", "")
            else:
                # Plain text response
                response = completion if isinstance(completion, str) else str(completion)
            
            # Get reward for this pair
            reward = self.get_reward(system_prompt, user_query, response)
            results.append(reward)
            
        return results

# Phi-3 uses a specific chat template with im_start/im_sep/im_end tokens
def format_phi_prompt(system_prompt, user_query):
    """Format the prompt according to Phi-3's chat template"""
    formatted_prompt = f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n<|im_start|>user<|im_sep|>\n{user_query}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n"
    return formatted_prompt

def load_and_prepare_dataset(path_or_data, train_size=0.95, cache_dir="./dataset_cache"):
    """
    Load and prepare dataset with system prompts and user queries
    
    Returns:
        train_dataset, eval_dataset: Datasets with 'prompt' column containing dictionaries
        with 'system_prompt' and 'user_query' keys
    """
    # Load dataset
    if isinstance(path_or_data, str) and os.path.exists(path_or_data):
        logger.info(f"Loading dataset from {path_or_data}")
        df = pd.read_csv(path_or_data)
    elif isinstance(path_or_data, pd.DataFrame):
        logger.info("Using provided DataFrame as dataset")
        df = path_or_data
    else:
        logger.warning("No dataset provided. Creating a sample dataset for demonstration...")
        df = pd.DataFrame({
            'system_prompt': [
                "You are a helpful AI assistant that provides accurate and concise information.",
                "You are an expert in programming who helps solve coding problems efficiently.",
                "You are a creative writer who creates engaging stories."
            ] * 10,
            'user_query': [
                "What are the main causes of climate change?",
                "How do I implement a binary search tree in Python?",
                "Write a short story about a robot discovering emotions."
            ] * 10
        })
    
    logger.info(f"Dataset size: {len(df)}")
    
    # Ensure we have the required columns
    required_columns = ['system_prompt', 'user_query']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Convert system prompts and user queries to the expected format
    # GRPO expects a 'prompt' column containing the complete prompt
    df['prompt'] = df.apply(
        lambda row: {
            'system_prompt': row['system_prompt'],
            'user_query': row['user_query']
        }, 
        axis=1
    )
    
    # Keep only the prompt column
    df = df[['prompt']]
    
    # Convert to HF dataset
    dataset = Dataset.from_pandas(df)
    
    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(eval_dataset)}")
    
    return dataset

# Command to start vLLM server (for reference, not called in the script)
def get_vllm_server_command(model_path="microsoft/Phi-3-mini-4k-instruct"):
    """
    Returns the command to start a vLLM server as a string.
    This is just for reference - you'll run this command separately.
    """
    cmd = (
        f"CUDA_VISIBLE_DEVICES=0,1,2 trl vllm-serve "
        f"--model {model_path} "
        f"--port {VLLM_PORT} "
        f"--enable-prefix-caching "
        f"--gpu-memory-utilization 0.90 "
        f"--max-model-len 4096 "
        f"--dtype half "
        f"--tensor-parallel-size 3"
    )
    return cmd

def main():
    # Print vLLM command for reference
    logger.info(f"To start the vLLM server in a separate terminal, run:")
    logger.info(get_vllm_server_command())
    logger.info("Make sure the vLLM server is running before starting this script!")
    
    # Load base model with quantization
    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Important: Disable model caching when using gradient checkpointing
    )

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    logger.info("Added LoRA adapters to model")
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Set up reward model with caching
    reward_model = GPT4oRewardModel(cache_dir="./reward_cache")
    
    # Load and prepare dataset (replace with your actual dataset path)
    train_dataset = load_and_prepare_dataset("your_dataset_path.csv") 
    
    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        # Pass only the batch_evaluate method as the reward function
        reward_funcs=reward_model.batch_evaluate,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Start training
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    logger.info(f"Training complete! Model saved to {OUTPUT_DIR}")
    
    # Training is complete
    logger.info("Training complete!")
    logger.info("Remember to stop your vLLM server when you're done.")

if __name__ == "__main__":
    main()
