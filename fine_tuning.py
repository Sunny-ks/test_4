import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from transformers import Phi4ForCausalLM, Phi4ImageProcessor, AutoTokenizer
from datasets import Dataset, load_dataset
import evaluate
from huggingface_hub import login
import json
import logging
from tqdm import tqdm
import argparse
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Moderation categories from the images
CATEGORIES = [
    "PORNOGRAPHY", "GORE", "WEAPONS", "EXTREMISM", "DRUGS", 
    "GAMBLING", "ID_CARDS", "DOCUMENTS", "FACE", 
    "SWIM_UNDERWEAR", "ALCOHOL", "Safe Content"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-4 Multimodal for image classification")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-4-vision", help="Model name or path")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./phi4-image-classifier", help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--log_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="phi4-image-classification", help="W&B project name")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token for model access")
    return parser.parse_args()

def prepare_prompt(image_path: str) -> str:
    """Prepare the prompt for image classification"""
    prompt = """You are an expert content moderation assistant. Your task is to 
classify the content of an image into one or more predefined categories. Use 
the detailed definitions below to decide which labels apply. If none of the 
categories apply, return Safe Content.

Categories and Definitions:

Label: PORNOGRAPHY
- Images: Content containing commercial/amateur pornography, CSAM, sexting 
selfies, nudity, sex acts, greyscale pornographic images, sexually explicit 
cartoons/manga
- Text: Explicit sexual content, erotic descriptions, sexually suggestive 
language

Label: GORE
- Images: Content showing gore, graphic violence, accident/shooting victims, 
beatings, mutilation, decapitation, blood/guts, self-harm, suicide
- Text: Descriptions of graphic violence, gore, self-harm, or disturbing 
injuries

Label: WEAPONS
- Images: Content showing rifles, machine guns, handguns, grenade launchers, 
swords, knives, people holding weapons
- Text: Descriptions of weapons, threats involving weapons, promotion of weapon 
usage

Label: EXTREMISM
- Images: Terrorist militants, executions, propaganda, acts of terrorism, KKK 
rallies, Nazi/white supremacy symbols, Hitler imagery
- Text: Extremist content, hate speech, terrorist/extremist propaganda

Label: DRUGS
- Images: Illegal/legal drugs, drug use, drug paraphernalia, drug-related 
plants and symbols
- Text: References to drug use/dealing, promotion of substances

Label: GAMBLING
- Images: Gambling scenes, casinos, lottery, betting, slots, card games, 
gambling advertisements
- Text: Gambling-related content, betting terminology, gambling promotion

Label: ID_CARDS
- Images: Passports, licenses, photo IDs, insurance cards, credit/debit cards, 
sensitive documents
- Text: Personal identification information, card numbers

Label: DOCUMENTS
- Images: Confidential business documents, invoices, financial statements, 
spreadsheets
- Text: Sensitive business information, confidential data

Label: FACE
- Images: Human faces in images/video
- Text: Descriptions identifying specific individuals

Label: SWIM_UNDERWEAR
- Images: People in swimwear, underwear, lingerie, beachwear
- Text: Descriptions of people in revealing but non-pornographic attire

Label: ALCOHOL
- Images: Alcoholic beverages, drinking scenes, bars, parties with alcohol, 
beer pong
- Text: References to alcohol consumption, drinking culture

Instructions:
- Analyze the image content carefully
- Return all applicable labels, separated by commas
- If none apply, return: Safe Content

Response Format:
PORNOGRAPHY, GORE, WEAPONS, EXTREMISM, DRUGS, GAMBLING, ID_CARDS, DOCUMENTS, 
FACE, SWIM_UNDERWEAR, ALCOHOL, or Safe Content

Note: Return only the label(s), without any additional explanation."""
    
    return prompt

def prepare_dataset(data_path: str, image_processor, tokenizer, max_length: int):
    """Prepare the dataset for training/validation"""
    # Load the dataset from JSON file or directory
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        dataset = Dataset.from_dict({
            'image_path': [item['image_path'] for item in data],
            'labels': [item['labels'] for item in data]
        })
    else:
        # Assume data_path is a directory with subdirectories for each class
        image_paths = []
        labels = []
        for category in os.listdir(data_path):
            category_path = os.path.join(data_path, category)
            if os.path.isdir(category_path):
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(category_path, img_file))
                        labels.append(category)
        dataset = Dataset.from_dict({
            'image_path': image_paths,
            'labels': labels
        })
    
    def preprocess_function(examples):
        images = [Image.open(path).convert('RGB') for path in examples['image_path']]
        image_inputs = image_processor(images, return_tensors="pt")
        
        prompts = [prepare_prompt(path) for path in examples['image_path']]
        text_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length)
        
        # Convert labels to format expected by model
        target_texts = examples['labels']
        target_encodings = tokenizer(target_texts, padding="max_length", truncation=True, max_length=128)
        
        return {
            "pixel_values": image_inputs.pixel_values,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": target_encodings.input_ids,
        }
    
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )
    
    return processed_dataset

class Phi4MultimodalClassifier:
    def __init__(self, model_name: str, hf_token: str = None):
        """Initialize the Phi-4 Multimodal classifier"""
        if hf_token:
            login(token=hf_token)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_processor = Phi4ImageProcessor.from_pretrained(model_name)
        self.model = Phi4ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Set special tokens
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded: {model_name}")
    
    def train(self, 
              train_dataset,
              val_dataset,
              output_dir: str,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-6,
              gradient_accumulation_steps: int = 4,
              log_wandb: bool = False,
              wandb_project: str = "phi4-image-classification"):
        """Fine-tune the model on the provided dataset"""
        from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
        
        if log_wandb:
            wandb.init(project=wandb_project)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=0.01,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="wandb" if log_wandb else "none",
            fp16=torch.cuda.is_available(),
        )
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        return trainer
    
    def predict(self, image_path: str) -> Tuple[str, Dict[str, float]]:
        """Predict the content moderation labels for a given image"""
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        
        # Prepare the prompt
        prompt = prepare_prompt(image_path)
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Combine inputs for prediction
        inputs = {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "pixel_values": image_inputs.pixel_values
        }
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode the output
        predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the prediction from the response
        response_lines = predicted_text.split("\n")
        prediction = None
        for line in response_lines:
            # Look for the response after the prompt
            if any(category in line for category in CATEGORIES):
                prediction = line.strip()
                break
        
        if not prediction:
            prediction = "Safe Content"  # Default if no prediction found
        
        # Calculate confidence scores (placeholder - in a real system you'd use logits)
        confidence_scores = {}
        predicted_categories = [cat.strip() for cat in prediction.split(",")]
        for category in CATEGORIES:
            confidence_scores[category] = 1.0 if category in predicted_categories else 0.0
        
        return prediction, confidence_scores
    
    def evaluate_model(self, test_dataset_path: str) -> Dict[str, float]:
        """Evaluate the model on a test dataset"""
        with open(test_dataset_path, 'r') as f:
            test_data = json.load(f)
        
        correct = 0
        total = len(test_data)
        results = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            image_path = item['image_path']
            true_labels = item['labels']
            
            prediction, _ = self.predict(image_path)
            predicted_labels = prediction if prediction != "Safe Content" else ""
            
            # Compare predictions with true labels
            predicted_set = set(label.strip() for label in predicted_labels.split(',') if label.strip())
            true_set = set(label.strip() for label in true_labels.split(',') if label.strip())
            
            is_correct = predicted_set == true_set
            if is_correct:
                correct += 1
            
            results.append({
                "image_path": image_path,
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "is_correct": is_correct
            })
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-category metrics
        category_metrics = {}
        for category in CATEGORIES:
            tp = fp = fn = 0
            for item in results:
                true_set = set(label.strip() for label in item["true_labels"].split(',') if label.strip())
                pred_set = set(label.strip() for label in item["predicted_labels"].split(',') if label.strip())
                
                if category in true_set and category in pred_set:
                    tp += 1
                elif category not in true_set and category in pred_set:
                    fp += 1
                elif category in true_set and category not in pred_set:
                    fn += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            category_metrics[category] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        evaluation_results = {
            "accuracy": accuracy,
            "category_metrics": category_metrics,
            "detailed_results": results
        }
        
        return evaluation_results

def main():
    args = parse_args()
    
    # Initialize the classifier
    classifier = Phi4MultimodalClassifier(args.model_name, args.hf_token)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = prepare_dataset(
        args.train_data, 
        classifier.image_processor, 
        classifier.tokenizer, 
        args.max_length
    )
    val_dataset = prepare_dataset(
        args.val_data, 
        classifier.image_processor, 
        classifier.tokenizer, 
        args.max_length
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_wandb=args.log_wandb,
        wandb_project=args.wandb_project
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
