import os
import argparse
import torch
from PIL import Image
import json
from tqdm import tqdm
from transformers import Phi4ForCausalLM, Phi4ImageProcessor, AutoTokenizer
from huggingface_hub import login
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Categories from the moderation system
CATEGORIES = [
    "PORNOGRAPHY", "GORE", "WEAPONS", "EXTREMISM", "DRUGS", 
    "GAMBLING", "ID_CARDS", "DOCUMENTS", "FACE", 
    "SWIM_UNDERWEAR", "ALCOHOL", "Safe Content"
]

# Mapping of your dataset folders to moderation categories - same as in data prep
FOLDER_TO_CATEGORY = {
    "alcohol": "ALCOHOL",
    "credit_card": "ID_CARDS",
    "drug-name": "DRUGS", 
    "gore": "GORE",
    "hate": "EXTREMISM",
    "memes": "Safe Content",
    "middle_finger": "Safe Content",
    "porn": "PORNOGRAPHY",
    "safe": "Safe Content",
    "self_harm": "GORE",
    "selfie": "FACE",
    "violence": "GORE",
    "weapons": "WEAPONS"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Phi-4 for image classification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model or HF model name")
    parser.add_argument("--input", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON file for predictions")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token for model access")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    return parser.parse_args()

def prepare_prompt(image_path):
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

class Phi4Classifier:
    def __init__(self, model_path, hf_token=None):
        """Initialize the model for inference"""
        if hf_token:
            login(token=hf_token)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model components
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.image_processor = Phi4ImageProcessor.from_pretrained(model_path)
        self.model = Phi4ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Set special tokens
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded: {model_path}")
    
    def predict_single(self, image_path):
        """Predict for a single image"""
        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            return "ERROR", {}
        
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
    
    def predict_batch(self, image_paths, batch_size=4):
        """Predict for a batch of images"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for path in batch_paths:
                try:
                    prediction, confidence = self.predict_single(path)
                    batch_results.append({
                        "image_path": path,
                        "prediction": prediction,
                        "confidence": confidence
                    })
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    batch_results.append({
                        "image_path": path,
                        "prediction": "ERROR",
                        "confidence": {}
                    })
            
            results.extend(batch_results)
        
        return results

def main():
    args = parse_args()
    
    # Initialize classifier
    classifier = Phi4Classifier(args.model_path, args.hf_token)
    
    # Determine input type (single image, directory, or JSON file)
    if os.path.isfile(args.input) and args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        # Single image
        image_paths = [args.input]
    elif os.path.isfile(args.input) and args.input.lower().endswith('.json'):
        # JSON file with image paths
        with open(args.input, 'r') as f:
            data = json.load(f)
        image_paths = [item['image_path'] for item in data]
    elif os.path.isdir(args.input):
        # Directory of images
        image_paths = []
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_paths.append(os.path.join(root, file))
    else:
        logger.error(f"Invalid input: {args.input}")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Run predictions
    results = classifier.predict_batch(image_paths, batch_size=args.batch_size)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Predictions saved to {args.output}")
    
    # Print summary
    predictions_by_category = {}
    for result in results:
        if result['prediction'] != "ERROR":
            categories = result['prediction'].split(", ")
            for category in categories:
                if category not in predictions_by_category:
                    predictions_by_category[category] = 0
                predictions_by_category[category] += 1
    
    logger.info("Prediction summary:")
    for category, count in sorted(predictions_by_category.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {category}: {count} images ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()
