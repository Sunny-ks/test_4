training_args = DPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    save_strategy="no",
    logging_steps=10,
    output_dir=ouput_dir,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,  # Changed from tokenizer to processing_class
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=12288,
    max_length=12800,
)
