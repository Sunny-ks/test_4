#!/usr/bin/env python3
import os
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Training parameters
MODEL_NAME = "Qwen/Qwen2-32B-Instruct"
DATASET_PATH = "your_data.csv"  # Replace with your CSV file path
OUTPUT_DIR = "./qwen-32b-finetuned"
SEED = 42
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 2  # Per device batch size (reduced for 32B model)
GRADIENT_ACCUMULATION_STEPS = 8  # Increased for 32B model
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
FINE_TUNING_TYPE = "qlora"  # Options: "full", "lora", "qlora"
LORA_RANK = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.05  # LoRA dropout
SAVE_TOTAL_LIMIT = 3

# Set seed for reproducibility
set_seed(SEED)

def prepare_inputs_and_labels(examples, tokenizer, max_length):
    """
    Prepare inputs with masked labels for SFT from CSV with 'question', 'prompt', and 'response' columns.
    Only compute loss on assistant responses.
    Uses Qwen-specific chat format.
    """
    conversations = []
    
    # Process each row from the CSV
    for i in range(len(examples['question'])):  # Assuming question, prompt, response are aligned
        system_prompt = examples['prompt'][i]
        user_question = examples['question'][i]
        assistant_response = examples['response'][i]
        
        # Format conversation with Qwen chat markers
        # Qwen uses <|im_start|>system/user/assistant<|im_end|> format
        full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        full_text += f"<|im_start|>user\n{user_question}<|im_end|>\n"
        full_text += f"<|im_start|>assistant\n{assistant_response}<|im_end|>\n"
        
        conversations.append(full_text)
    
    # Tokenize the conversations
    tokenized_inputs = tokenizer(
        conversations,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    input_ids = tokenized_inputs["input_ids"]
    labels = input_ids.clone()
    
    # Mask labels for non-assistant parts (only compute loss on assistant responses)
    for idx, text in enumerate(conversations):
        # Find the position of assistant response
        assistant_start = text.find("<|im_start|>assistant\n")
        if assistant_start == -1:
            continue  # Skip if no assistant part found
            
        # Get token positions before assistant part
        prefix = text[:assistant_start]
        prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        prefix_length = len(prefix_tokens)
        
        # Mask all tokens before the assistant part with -100
        if prefix_length < max_length:
            labels[idx, :prefix_length] = -100
            
        # Also mask the assistant marker tokens
        marker = "<|im_start|>assistant\n"
        marker_tokens = tokenizer(marker, add_special_tokens=False)["input_ids"]
        marker_length = len(marker_tokens)
        
        # Mask the tokens for the assistant marker
        if prefix_length + marker_length < max_length:
            labels[idx, prefix_length:prefix_length + marker_length] = -100
            
        # Also mask the end token of the assistant response
        end_marker = "<|im_end|>"
        end_marker_pos = text.find(end_marker, assistant_start)
        if end_marker_pos != -1:
            end_tokens = tokenizer(text[:end_marker_pos], add_special_tokens=False)["input_ids"]
            end_pos = len(end_tokens)
            if end_pos < max_length:
                labels[idx, end_pos:] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels,
    }

def setup_lora_or_qlora(model, fine_tuning_type="lora"):
    """Configure LoRA or QLoRA for efficient fine-tuning"""
    # Target modules for Qwen models
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    if fine_tuning_type == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["embed_tokens", "lm_head"],
        )
    elif fine_tuning_type == "qlora":
        # QLoRA config uses 4-bit quantization
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["embed_tokens", "lm_head"],
        )
    else:
        raise ValueError(f"Unsupported fine-tuning type: {fine_tuning_type}")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Determine local process index for device mapping
    local_rank = accelerator.local_process_index
    world_size = accelerator.num_processes
    
    logger.info(f"Process rank: {local_rank}, world size: {world_size}")
    
    # Load tokenizer
    logger.info("Loading Qwen tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
    )
    
    # Ensure the tokenizer has appropriate settings
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set up flash attention for faster training if possible
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Load model based on fine-tuning type
    logger.info(f"Loading model on device {local_rank} with fine-tuning type: {FINE_TUNING_TYPE}")
    
    if FINE_TUNING_TYPE == "qlora":
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # For QLoRA, we load the model differently
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": accelerator.local_process_index},
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply QLoRA
        model = setup_lora_or_qlora(model, "qlora")
    
    elif FINE_TUNING_TYPE == "lora":
        # Standard model loading for LoRA
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map={"": accelerator.local_process_index},
        )
        
        # Apply LoRA
        model = setup_lora_or_qlora(model, "lora")
    
    else:  # Full fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map={"": accelerator.local_process_index},
        )
    
    # Load CSV dataset
    logger.info("Loading CSV dataset")
    raw_datasets = load_dataset('csv', data_files=DATASET_PATH)
    
    # Check if required columns exist
    required_columns = ['question', 'prompt', 'response']
    for col in required_columns:
        if col not in raw_datasets['train'].column_names:
            raise ValueError(f"CSV file must contain '{col}' column")
    
    # Split dataset into train and validation if needed
    if 'validation' not in raw_datasets:
        logger.info("Creating validation split from training data")
        # Split train set into train and validation
        split_datasets = raw_datasets["train"].train_test_split(test_size=0.05, seed=SEED)
        raw_datasets = {
            "train": split_datasets["train"],
            "validation": split_datasets["test"]
        }
    
    # Preprocess the dataset
    logger.info("Preprocessing dataset")
    
    # Apply preprocessing to the dataset
    train_dataset = raw_datasets["train"].map(
        lambda examples: prepare_inputs_and_labels(examples, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing and masking train dataset",
    )
    
    eval_dataset = raw_datasets["validation"].map(
        lambda examples: prepare_inputs_and_labels(examples, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing and masking validation dataset",
    )
    
    # Set up optimizer
    optim = "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        evaluation_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        optim=optim,
        do_eval=True,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        remove_unused_columns=False,  # Important for custom data formats
        report_to="tensorboard",
        label_names=["labels"],  # Explicitly set label_names to fix the warning
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We want causal language modeling
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_names=["labels"],  # Set label_names to fix warning
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model (process 0 only)
    if accelerator.is_local_main_process:
        logger.info("Saving final model")
        trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()
