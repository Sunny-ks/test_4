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
    
    # Generation params
    temperature=0.8,                   # Slightly lower for more focused outputs
    top_p=0.92,                        # Nucleus sampling parameter
    top_k=50,                          # Top-k filtering
    max_prompt_length=384,             # Based on typical prompt length
    max_completion_length=128,         # Reasonable completion length
    repetition_penalty=1.1,            # Reduce repetition
    
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


# Initialize the GRPO trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_model.batch_evaluate,  # Use our GPT-4o reward model
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
