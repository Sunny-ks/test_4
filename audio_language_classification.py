import os
import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import librosa
from datasets import load_dataset, Audio, concatenate_datasets, Dataset
from transformers import (
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2ForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb  # Optional for tracking experiments
from collections import Counter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Configuration
class Config:
    model_name = "facebook/wav2vec2-xls-r-300m"
    
    # All languages to train on as distinct classes
    languages = [
        "en",   # English
        "es",   # Spanish
        "it",   # Italian
        "hi",   # Hindi
        "zh",   # Chinese (Mandarin)
        "tl",   # Tagalog/Filipino
        "pt",   # Portuguese
        "fr",   # French
        "de",   # German
        "pa",   # Punjabi
        "ru",   # Russian
        "uk",   # Ukrainian
        "ar",   # Arabic
        "ja",   # Japanese
        "ko",   # Korean
        "tr",   # Turkish
        "nl",   # Dutch
        "sv",   # Swedish
        "fi",   # Finnish
        "pl",   # Polish
        "cs",   # Czech
        "hu",   # Hungarian
        "el",   # Greek
        "vi",   # Vietnamese
        "id",   # Indonesian
        "th",   # Thai
        "fa",   # Persian/Farsi
        "he",   # Hebrew
    ]
    
    samples_per_language = 1000  # Limit samples per language for faster experimentation
    samples_per_noise = 1000     # Number of noise samples to use
    
    max_duration = 5.0  # Maximum audio duration in seconds
    target_sampling_rate = 16000  # Target sampling rate for all audio
    batch_size = 8
    learning_rate = 3e-5
    weight_decay = 0.01
    num_train_epochs = 10
    warmup_ratio = 0.1
    output_dir = "./wav2vec2-language-id"
    push_to_hub = False  # Set to True if you want to upload to Hugging Face Hub
    hub_model_id = "wav2vec2-xls-r-300m-language-id"  # Your model ID on HF Hub
    hub_token = None  # Your HF token if pushing to Hub
    # If a language fails to load, skip it instead of failing completely
    skip_failed_languages = True

config = Config()

# For logging (optional)
# wandb.init(project="language-id", name="wav2vec2-xls-r-300m-language-id")

print("Loading datasets...")

# Function to load VoxLingua107 for specific languages
def load_voxlingua(languages, samples_per_language):
    all_datasets = []
    successful_languages = []
    
    for i, lang in enumerate(languages):
        try:
            # Try to load directly from Hugging Face
            ds = load_dataset("facebook/voxlingua107", lang, split="train")
            
            # Add language label and language string
            ds = ds.map(lambda example: {
                "language": i, 
                "language_str": lang,
                "original_lang": lang
            })
            
            # Limit number of samples
            if len(ds) > samples_per_language:
                ds = ds.select(range(samples_per_language))
                
            # Add audio column with correct sampling rate
            ds = ds.cast_column("audio", Audio(sampling_rate=config.target_sampling_rate))
            
            all_datasets.append(ds)
            successful_languages.append(lang)
            print(f"Loaded {len(ds)} samples for {lang}")
        except Exception as e:
            print(f"Error loading {lang}: {e}")
            if not config.skip_failed_languages:
                raise
    
    return all_datasets, successful_languages

# Load MUSAN noise dataset
def load_musan_noise(samples_per_noise):
    try:
        noise_dataset = load_dataset("musan", "noise", split="train")
        
        # Noise will get its own label at the end
        noise_label = -1  # Temporary placeholder, will be updated later
        noise_dataset = noise_dataset.map(lambda example: {
            "language": noise_label, 
            "language_str": "noise",
            "original_lang": "noise"
        })
        
        # Limit samples
        if len(noise_dataset) > samples_per_noise:
            noise_dataset = noise_dataset.select(range(samples_per_noise))
            
        # Convert audio to target sampling rate
        noise_dataset = noise_dataset.cast_column("audio", Audio(sampling_rate=config.target_sampling_rate))
        
        print(f"Loaded {len(noise_dataset)} noise samples")
        return noise_dataset
    except Exception as e:
        print(f"Error loading MUSAN noise: {e}")
        return None

# Alternative MUSAN noise loading function (if the above fails)
def load_musan_noise_alternative(samples_per_noise):
    try:
        # Try to load with a different approach
        noise_dataset = load_dataset("musan", split="train.noise")
        
        # Noise label (will be updated later)
        noise_label = -1  # Temporary placeholder
        noise_dataset = noise_dataset.map(lambda example: {
            "language": noise_label, 
            "language_str": "noise",
            "original_lang": "noise"
        })
        
        # Limit samples
        if len(noise_dataset) > samples_per_noise:
            noise_dataset = noise_dataset.select(range(samples_per_noise))
            
        # Convert audio to target sampling rate
        noise_dataset = noise_dataset.cast_column("audio", Audio(sampling_rate=config.target_sampling_rate))
        
        print(f"Loaded {len(noise_dataset)} noise samples via alternative method")
        return noise_dataset
    except Exception as e:
        print(f"Error loading MUSAN noise via alternative method: {e}")
        
        # Create a synthetic noise dataset as fallback
        print("Creating synthetic noise dataset as fallback")
        
        # Noise label (will be updated later)
        noise_label = -1  # Temporary placeholder
        
        samples = []
        for i in range(samples_per_noise):
            # Generate white noise
            noise = np.random.normal(0, 0.1, int(config.target_sampling_rate * config.max_duration))
            samples.append({
                "audio": {"array": noise, "sampling_rate": config.target_sampling_rate},
                "language": noise_label,
                "language_str": "noise",
                "original_lang": "noise"
            })
        
        noise_dataset = Dataset.from_dict({
            "audio": [s["audio"] for s in samples],
            "language": [s["language"] for s in samples],
            "language_str": [s["language_str"] for s in samples],
            "original_lang": [s["original_lang"] for s in samples]
        })
        
        # Cast audio column
        noise_dataset = noise_dataset.cast_column("audio", Audio(sampling_rate=config.target_sampling_rate))
        
        print(f"Created {len(noise_dataset)} synthetic noise samples")
        return noise_dataset

# Load language datasets
language_datasets, successful_languages = load_voxlingua(
    config.languages, 
    config.samples_per_language
)
print(f"Successfully loaded {len(successful_languages)}/{len(config.languages)} languages")

# Load noise dataset
noise_dataset = load_musan_noise(config.samples_per_noise)

# Try alternative method if first method fails
if noise_dataset is None:
    noise_dataset = load_musan_noise_alternative(config.samples_per_noise)

# Update the languages list to only include successfully loaded languages
config.languages = successful_languages

# Combine all datasets and assign final class IDs
# First, prepare the list of all datasets
all_datasets = []

# 1. Add languages (with their original indices)
for i, dataset in enumerate(language_datasets):
    # Set the correct language ID
    dataset = dataset.map(lambda example: {"language": i})
    all_datasets.append(dataset)

# 2. Add noise (with index after all languages)
noise_id = len(successful_languages)
if noise_dataset is not None:
    # Set the noise language ID
    noise_dataset = noise_dataset.map(lambda example: {"language": noise_id})
    all_datasets.append(noise_dataset)

# Combine all datasets
combined_dataset = concatenate_datasets(all_datasets)

# Split into train, validation, test
train_test_split = combined_dataset.train_test_split(test_size=0.2, seed=seed)
test_val_split = train_test_split["test"].train_test_split(test_size=0.5, seed=seed)

train_dataset = train_test_split["train"]
val_dataset = test_val_split["train"]
test_dataset = test_val_split["test"]

print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# Check class distribution
label_counts = Counter(train_dataset["language"])
print("\nClass distribution in training set:")
for label, count in sorted(label_counts.items()):
    label_name = "noise" if label == noise_id else config.languages[label]
    print(f"  {label_name}: {count} samples")

# Create a label to language mapping for later use
id2label = {}
label2id = {}

# Add languages
for i, lang in enumerate(config.languages):
    id2label[i] = lang
    label2id[lang] = i

# Add noise category
id2label[noise_id] = "noise"
label2id["noise"] = noise_id

# Add "others" category for inference (not included in training)
others_id = noise_id + 1
id2label[others_id] = "others"
label2id["others"] = others_id

print("\nLanguage to ID mapping:")
for id, lang in sorted(id2label.items()):
    print(f"  {id}: {lang}")

# Load feature extractor and model
print("\nLoading model and feature extractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)

# Initialize model with number of classes (languages + noise + others)
# Note: The "others" class is included in the model but won't be used during training
num_labels = len(id2label)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    config.model_name,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

print(f"Model loaded with {num_labels} classification labels (including 'others' class for inference)")

# Audio augmentation function (to improve model robustness)
def augment_audio(audio_array, sampling_rate):
    # Randomly decide whether to apply augmentation
    if np.random.random() < 0.5:
        return audio_array
    
    # Choose a random augmentation method
    aug_type = np.random.choice(['time_shift', 'speed', 'noise', 'all'])
    
    if aug_type in ['time_shift', 'all']:
        # Time shift: shift the audio by a random amount
        shift_factor = np.random.uniform(-0.2, 0.2)
        shift_amount = int(len(audio_array) * shift_factor)
        if shift_amount > 0:
            audio_array = np.pad(audio_array, (shift_amount, 0))[:len(audio_array)]
        elif shift_amount < 0:
            audio_array = np.pad(audio_array, (0, -shift_amount))[(-shift_amount):]
    
    if aug_type in ['speed', 'all']:
        # Speed change: change the speed of the audio
        speed_factor = np.random.uniform(0.9, 1.1)
        
        # Use librosa for speed change
        try:
            # Store the original length
            original_length = len(audio_array)
            audio_array = librosa.effects.time_stretch(audio_array, rate=speed_factor)
            
            # Restore original length
            if len(audio_array) > original_length:
                audio_array = audio_array[:original_length]
            elif len(audio_array) < original_length:
                audio_array = np.pad(audio_array, (0, original_length - len(audio_array)))
        except:
            # Fallback if librosa fails
            pass
    
    if aug_type in ['noise', 'all']:
        # Add slight background noise
        noise_factor = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_factor, len(audio_array))
        audio_array = audio_array + noise
        
        # Clip to prevent overflow
        audio_array = np.clip(audio_array, -1.0, 1.0)
    
    return audio_array

# Preprocess function - prepare audio for model
def preprocess_function(examples, augment=False):
    # Load and resample audio data
    audio_arrays = []
    for audio in examples["audio"]:
        # Get audio array
        audio_array = audio["array"]
        
        # Apply augmentation if requested (only for training data)
        if augment:
            audio_array = augment_audio(audio_array, config.target_sampling_rate)
        
        # Cut or pad to max_duration
        target_length = int(config.max_duration * config.target_sampling_rate)
        
        if len(audio_array) > target_length:
            # Cut to max_duration
            audio_array = audio_array[:target_length]
        elif len(audio_array) < target_length:
            # Use symmetric padding instead of one-sided padding
            padding_needed = target_length - len(audio_array)
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            audio_array = np.pad(audio_array, (pad_left, pad_right), mode='constant')
        
        audio_arrays.append(audio_array)
    
    # Prepare inputs for the model
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=config.target_sampling_rate,
        padding=True,
        return_tensors="pt",
    )
    
    # Add labels
    inputs["labels"] = examples["language"]
    
    return inputs

# Apply preprocessing to all datasets - with augmentation for training data
print("Preprocessing datasets...")
train_dataset = train_dataset.map(
    lambda examples: preprocess_function(examples, augment=True), 
    batched=True, 
    remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(
    lambda examples: preprocess_function(examples, augment=False), 
    batched=True, 
    remove_columns=val_dataset.column_names
)
test_dataset = test_dataset.map(
    lambda examples: preprocess_function(examples, augment=False), 
    batched=True, 
    remove_columns=test_dataset.column_names
)

# Define compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted")
    }
    
    return metrics

# Prepare training arguments
training_args = TrainingArguments(
    output_dir=config.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=config.num_train_epochs,
    weight_decay=config.weight_decay,
    warmup_ratio=config.warmup_ratio,
    push_to_hub=config.push_to_hub,
    hub_model_id=config.hub_model_id,
    hub_token=config.hub_token,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Use mixed precision when available
    report_to="wandb" if "wandb" in globals() and wandb.run is not None else "none",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test results: {test_results}")

# Compute per-language metrics
def compute_per_language_metrics(dataset):
    print("\nPer-language evaluation:")
    
    trainer.model.eval()
    trainer.model.to(device)  # Ensure model is on the correct device
    
    all_preds = []
    all_labels = []
    
    # Collect predictions
    for batch in trainer.get_eval_dataloader(dataset):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = trainer.model(**batch)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # Calculate metrics per language
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    results = {}
    # Only evaluate on classes that were in the training data (not "others")
    for lang_id, lang_name in id2label.items():
        if lang_id == others_id:  # Skip "others" class which wasn't in training
            continue
            
        mask = all_labels == lang_id
        if np.sum(mask) == 0:
            continue
            
        lang_preds = all_preds[mask]
        lang_labels = all_labels[mask]
        
        accuracy = accuracy_score(lang_labels, lang_preds)
        results[lang_name] = {
            "accuracy": accuracy,
            "count": np.sum(mask)
        }
        
        print(f"  {lang_name}: Accuracy = {accuracy:.4f}, Count = {np.sum(mask)}")
    
    return results

per_language_metrics = compute_per_language_metrics(test_dataset)

# Save the model and feature extractor
trainer.save_model(config.output_dir)
feature_extractor.save_pretrained(config.output_dir)

# Save the language mapping
with open(os.path.join(config.output_dir, "language_mapping.txt"), "w") as f:
    f.write("ID to Language Mapping:\n")
    for id, label in id2label.items():
        f.write(f"{id}: {label}\n")

# Save test metrics
with open(os.path.join(config.output_dir, "test_metrics.txt"), "w") as f:
    f.write("Overall metrics:\n")
    for metric, value in test_results.items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nPer-language metrics:\n")
    for lang, metrics in per_language_metrics.items():
        f.write(f"{lang}: Accuracy = {metrics['accuracy']}, Count = {metrics['count']}\n")

# Create inference function that handles "others" class
def predict_language(audio_file_path, threshold=0.5):
    """
    Predict language from audio file with "others" detection.
    
    Args:
        audio_file_path: Path to audio file
        threshold: Confidence threshold below which to classify as "others"
        
    Returns:
        Dictionary with predicted language and confidence scores
    """
    # Load model if not already loaded
    model.eval()
    
    # Load audio file
    audio, sr = librosa.load(audio_file_path, sr=config.target_sampling_rate)
    
    # Truncate or pad to max_duration
    target_length = int(config.max_duration * config.target_sampling_rate)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Use symmetric padding
        padding_needed = target_length - len(audio)
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    
    # Process with feature extractor
    inputs = feature_extractor(
        audio, 
        sampling_rate=config.target_sampling_rate, 
        return_tensors="pt"
    ).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Remove the "others" class from prediction since it's only for inference
        # We'll determine if it's "others" based on confidence
        filtered_logits = logits[0, :-1]  # All except the last class ("others")
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        
        # Get the most likely class and its probability
        max_prob, predicted_class_id = torch.max(probabilities, dim=-1)
        
        # If max probability is below threshold, classify as "others"
        if max_prob.item() < threshold:
            predicted_label = "others"
            predicted_class_id = others_id
        else:
            predicted_label = id2label[predicted_class_id.item()]
    
    # Get all confidence scores (excluding "others" class)
    scores = probabilities.tolist()
    predictions = {id2label[i]: score for i, score in enumerate(scores)}
    
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "language": predicted_label,
        "confidence": max_prob.item(),
        "scores": dict(sorted_predictions[:3])  # Top 3 predictions
    }

# Save a sample inference script that includes "others" detection
with open(os.path.join(config.output_dir, "inference.py"), "w") as f:
    f.write("""
import os
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

def load_model(model_dir):
    # Load feature extractor and model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    
    # Get language mapping
    id2label = model.config.id2label
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, feature_extractor, id2label, device

def predict_language(audio_file_path, model, feature_extractor, id2label, device, threshold=0.5, max_duration=5.0, target_sampling_rate=16000):
    """
    Predict language from audio file with "others" detection.
    
    Args:
        audio_file_path: Path to audio file
        model: Loaded model
        feature_extractor: Feature extractor
        id2label: ID to language mapping
        device: Device to run inference on
        threshold: Confidence threshold below which to classify as "others"
        max_duration: Maximum audio duration to process
        target_sampling_rate: Target sampling rate for audio
        
    Returns:
        Dictionary with predicted language and confidence scores
    """
    # Load audio file
    audio, sr = librosa.load(audio_file_path, sr=target_sampling_rate)
    
    # Truncate or pad to max_duration
    target_length = int(max_duration * target_sampling_rate)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Use symmetric padding
        padding_needed = target_length - len(audio)
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    
    # Process with feature extractor
    inputs = feature_extractor(
        audio, 
        sampling_rate=target_sampling_rate, 
        return_tensors="pt"
    ).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Determine if an "others" class is present at the end
        # This depends on how the model was trained
        has_others_class = "others" in id2label.values()
        others_id = None
        
        # Find the ID for "others" class if it exists
        for id_val, label in id2label.items():
            if label == "others":
                others_id = int(id_val) if isinstance(id_val, str) else id_val
                break
        
        # If there's an explicit "others" class, remove it from prediction logits
        if has_others_class and others_id is not None:
            # Use all logits except the "others" class
            filtered_indices = [i for i in range(logits.shape[1]) if i != others_id]
            filtered_logits = logits[0, filtered_indices]
        else:
            filtered_logits = logits[0]
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        
        # Get the most likely class and its probability
        max_prob, max_idx = torch.max(probabilities, dim=-1)
        
        # Map max_idx back to original index if we filtered out "others"
        if has_others_class and others_id is not None:
            predicted_class_id = filtered_indices[max_idx.item()]
        else:
            predicted_class_id = max_idx.item()
        
        # If max probability is below threshold, classify as "others"
        if max_prob.item() < threshold:
            predicted_label = "others"
            predicted_class_id = others_id if others_id is not None else -1
        else:
            # Handle the case where id2label keys are strings
            pred_id = str(predicted_class_id) if str(predicted_class_id) in id2label else predicted_class_id
            predicted_label = id2label[pred_id]
    
    # Get all confidence scores
    # Exclude "others" class from scores (if it exists)
    scores = {}
    for i, score in enumerate(probabilities.tolist()):
        # Map index back to original index if we filtered out "others"
        if has_others_class and others_id is not None:
            orig_idx = filtered_indices[i]
        else:
            orig_idx = i
            
        # Handle the case where id2label keys are strings
        idx_key = str(orig_idx) if str(orig_idx) in id2label else orig_idx
        
        if idx_key in id2label:
            scores[id2label[idx_key]] = score
    
    sorted_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "language": predicted_label,
        "confidence": max_prob.item(),
        "scores": dict(sorted_predictions[:3])  # Top 3 predictions
    }

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict language from audio file")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--model_dir", default=".", help="Directory containing the model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for 'others' classification")
    
    args = parser.parse_args()
    
    model, feature_extractor, id2label, device = load_model(args.model_dir)
    result = predict_language(args.audio_file, model, feature_extractor, id2label, device, threshold=args.threshold)
    
    print(f"Predicted language: {result['language']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Top 3 scores:")
    for lang, score in result['scores'].items():
        print(f"  {lang}: {score:.4f}")
""")

print("Training and evaluation complete!")
print(f"Model saved to: {config.output_dir}")
print("You can use the saved inference.py script for language identification.")
print("\nThe model includes an 'others' category that will be assigned when:")
print("1. The confidence in any known language is below the threshold")
print("2. The audio contains a language not seen during training")
print("\nYou can adjust the confidence threshold when using the inference script.")
