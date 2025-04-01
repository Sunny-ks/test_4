def load_and_prepare_dataset(path_or_data, train_size=0.95, cache_dir="./dataset_cache"):
    """
    Load and prepare dataset with system prompts and user queries
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
    
    # Convert to HF dataset
    dataset = Dataset.from_pandas(df)
    
    # Split into train and validation sets
    splits = dataset.train_test_split(test_size=1-train_size, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    
    logger.info(f"Train set size: {len(train_dataset)}, Validation set size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


    train_dataset, eval_dataset = load_and_prepare_dataset("your_dataset_path.csv") 
