### pip install dspy-ai==2.3.0

import dspy
import re
import torch
import pandas as pd
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split

# Custom LM wrapper for Phi-4 with stop token handling
class Phi4LabelPredictor(dspy.LM):
    def __init__(self, model_name="microsoft/Phi-4", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.stop_token_id = self.tokenizer.encode("@", add_special_tokens=False)[0]
    
    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        max_tokens = kwargs.get("max_tokens", 100)
        temperature = kwargs.get("temperature", 0.3)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                eos_token_id=self.stop_token_id
            )
        
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        if response.endswith("@"):
            response = response[:-1]
            
        return response.strip()

# Custom example class for your dataset structure
class LabelClassificationExample(dspy.Example):
    system_prompt: str    # The "prompt" column from your CSV
    user_query: str       # The "question" column from your CSV
    expected_response: str  # The "response" column from your CSV
    
    # Derived field for expected labels (extracted from response)
    @property
    def expected_labels(self):
        label_match = re.search(r'\[([\d,\s]+)\]@', self.expected_response)
        if label_match:
            return [label.strip() for label in label_match.group(1).split(',')]
        elif "safe" in self.expected_response.lower():
            return ["safe"]
        else:
            return []

# Function to load and prepare examples from CSV
def load_examples_from_csv(csv_path):
    """Load examples directly from the CSV with prompt, response, question columns"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_columns = ["prompt", "response", "question"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns in CSV: {missing}")
        
        # Create examples from dataframe
        examples = []
        for _, row in df.iterrows():
            example = LabelClassificationExample(
                system_prompt=row["prompt"],
                user_query=row["question"],
                expected_response=row["response"]
            )
            examples.append(example)
        
        print(f"Loaded {len(examples)} examples from {csv_path}")
        
        # Show sample of the data
        if examples:
            sample = examples[0]
            print("\nSample example:")
            print(f"User query: {sample.user_query}")
            print(f"Expected response: {sample.expected_response}")
            print(f"Extracted labels: {sample.expected_labels}")
        
        return examples
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

# Module for label prediction
class LabelPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("context -> labels")
    
    def forward(self, system_prompt, user_query):
        # Construct the full prompt with Phi-4 chat format
        full_prompt = (
            f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user<|im_sep|>\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant<|im_sep|>\n"
        )
        
        return self.predictor(context=full_prompt)

# Label and format evaluation metric
def label_format_metric(example, prediction):
    # Extract predicted labels
    prediction_text = prediction.labels
    
    # Extract labels with regex
    label_match = re.search(r'\[([\d,\s]+)\]@', prediction_text)
    if not label_match:
        return 0.0
    
    predicted_labels = [label.strip() for label in label_match.group(1).split(',')]
    expected_labels = example.expected_labels
    
    # Format compliance check
    format_compliance = 1.0 if "confidence" in prediction_text and "%" in prediction_text else 0.0
    
    # First label precision
    first_precision = 1.0 if predicted_labels and expected_labels and predicted_labels[0] in expected_labels else 0.0
    
    # Overall label matching
    pred_set = set(predicted_labels)
    expected_set = set(expected_labels)
    
    set_precision = len(pred_set & expected_set) / len(pred_set) if pred_set else 0
    set_recall = len(pred_set & expected_set) / len(expected_set) if expected_set else 0
    
    f1 = 2 * set_precision * set_recall / (set_precision + set_recall) if (set_precision + set_recall) > 0 else 0
    
    # Combined score
    combined_score = format_compliance * (0.6 * first_precision + 0.4 * f1)
    
    return combined_score

# Function to save the optimized prompt to a text file
def save_optimized_prompt(prompt, output_dir="optimized_prompts"):
    """Save the optimized prompt to a text file with timestamp"""
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_prompt_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Write prompt to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        print(f"\nOptimized prompt saved to: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Error saving optimized prompt: {e}")
        return None

# Main optimization function
def optimize_phi4_prompts(csv_path, test_size=0.2, random_state=42, save_prompt=True):
    # Load examples from CSV
    examples = load_examples_from_csv(csv_path)
    if not examples:
        print("No examples loaded. Exiting.")
        return None
    
    # Split into training and evaluation sets
    train_examples, eval_examples = train_test_split(
        examples, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Split into {len(train_examples)} training and {len(eval_examples)} evaluation examples")
    
    # Initialize Phi-4 model
    phi4_lm = Phi4LabelPredictor()
    dspy.settings.configure(lm=phi4_lm)
    
    # Create predictor and teleprompter
    predictor = LabelPredictor()
    teleprompter = dspy.Teleprompter(
        predictor,
        metric=label_format_metric,
        max_iterations=10,
        optimize_for='quality'
    )
    
    # Run optimization
    print("Starting prompt optimization...")
    optimized_predictor = teleprompter.optimize(
        trainset=train_examples,
        valset=eval_examples,
        verbose=True
    )
    
    print("\nOptimization complete!")
    
    # Extract optimized system prompt
    optimized_system_prompt = optimized_predictor.predictor.prompt
    print("\nOptimized system prompt:")
    print(optimized_system_prompt)
    
    # Save the optimized prompt if requested
    if save_prompt:
        save_optimized_prompt(optimized_system_prompt)
    
    return optimized_predictor, optimized_system_prompt

# Function to test the optimized model
def test_optimized_model(model, test_queries, system_prompt):
    results = []
    
    print("\nTesting optimized model with sample queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        result = model(system_prompt=system_prompt, user_query=query)
        
        # Extract labels from the response
        label_match = re.search(r'\[([\d,\s]+)\]@', result.labels)
        if label_match:
            predicted_labels = [label.strip() for label in label_match.group(1).split(',')]
            print(f"Predicted labels: {predicted_labels}")
        else:
            print("No valid label format detected")
            
        print(f"Full prediction: {result.labels}")
        results.append(result.labels)
    
    return results

# Main execution function
if __name__ == "__main__":
    # Path to your CSV file
    csv_file = "your_dataset.csv"  # Replace with your actual file path
    
    # Run optimization and save prompt
    optimized_model, optimized_prompt = optimize_phi4_prompts(
        csv_file,
        save_prompt=True  # Set to True to save the prompt to a file
    )
    
    if optimized_model:
        # Test queries
        test_queries = [
            "I hate all immigrants and they should be deported.",
            "The weather is nice today and I'm enjoying the sunshine.",
            "Women should stay at home and not be allowed to work."
        ]
        
        # Test with the optimized prompt
        test_optimized_model(optimized_model, test_queries, optimized_prompt)
