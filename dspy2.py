CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model mosaicml/mpt-7b \
  --port 8000 \
  --max-model-len 122888 \
  --max-num-seqs 8 \
  --enable-prefix-caching



import dspy
import re
import torch
import pandas as pd
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import List

# Custom LM wrapper for Phi-4 using dspy.LM
class Phi4LabelPredictor(dspy.LM):
    def __init__(self, model_name="microsoft/Phi-4", device="cuda"):
        super().__init__(model=model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.stop_token_id = self.tokenizer.encode("@", add_special_tokens=False)[0]

    def basic_request(self, prompt, **kwargs):
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

# Define the signature for label classification
class LabelClassificationSignature(dspy.Signature):
    system_prompt: str = dspy.InputField()
    user_query: str = dspy.InputField()
    labels: List[str] = dspy.OutputField()

# Module for label prediction
class LabelPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(LabelClassificationSignature)

    def forward(self, system_prompt, user_query):
        return self.predictor(system_prompt=system_prompt, user_query=user_query)

# Function to load and prepare examples from CSV
def load_examples_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)

        required_columns = ["prompt", "response", "question"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns in CSV: {missing}")

        examples = []
        for _, row in df.iterrows():
            example = dspy.Example(
                system_prompt=row["prompt"],
                user_query=row["question"],
                labels=extract_labels(row["response"])
            )
            examples.append(example)

        print(f"Loaded {len(examples)} examples from {csv_path}")

        if examples:
            sample = examples[0]
            print("\nSample example:")
            print(f"User query: {sample.user_query}")
            print(f"Expected labels: {sample.labels}")

        return examples

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

# Extract labels from the response
def extract_labels(response):
    label_match = re.search(r'\\[([\\d,\\s]+)\\]@', response)
    if label_match:
        return [label.strip() for label in label_match.group(1).split(',')]
    elif "safe" in response.lower():
        return ["safe"]
    else:
        return []

# Label and format evaluation metric
def label_format_metric(example, pred):
    predicted_labels = pred.labels
    expected_labels = example.labels

    format_compliance = 1.0 if isinstance(predicted_labels, list) else 0.0
    first_precision = 1.0 if predicted_labels and expected_labels and predicted_labels[0] in expected_labels else 0.0

    pred_set = set(predicted_labels)
    expected_set = set(expected_labels)

    set_precision = len(pred_set & expected_set) / len(pred_set) if pred_set else 0
    set_recall = len(pred_set & expected_set) / len(expected_set) if expected_set else 0

    f1 = 2 * set_precision * set_recall / (set_precision + set_recall) if (set_precision + set_recall) > 0 else 0

    combined_score = format_compliance * (0.6 * first_precision + 0.4 * f1)

    return combined_score

# Function to save the optimized prompt to a text file
def save_optimized_prompt(prompt, output_dir="optimized_prompts"):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_prompt_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"\nOptimized prompt saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error saving optimized prompt: {e}")
        return None

# Main optimization function
def optimize_phi4_prompts(csv_path, test_size=0.2, random_state=42, save_prompt=True):
    examples = load_examples_from_csv(csv_path)
    if not examples:
        print("No examples loaded. Exiting.")
        return None, None

    train_examples, eval_examples = train_test_split(
        examples,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Split into {len(train_examples)} training and {len(eval_examples)} evaluation examples")

    phi4_lm = Phi4LabelPredictor()
    dspy.settings.configure(lm=phi4_lm)

    predictor = LabelPredictor()

    optimizer = dspy.MIPROv2(metric=label_format_metric)

    print("Starting prompt optimization...")
    optimized_predictor = optimizer.compile(
        predictor,
        trainset=train_examples,
        valset=eval_examples,
        num_rounds=10,
        num_candidates=3
    )

    print("\nOptimization complete!")

    optimized_system_prompt = optimized_predictor.predictor.prompt
    print("\nOptimized system prompt:")
    print(optimized_system_prompt)

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

        print(f"Predicted labels: {result.labels}")
        results.append(result.labels)

    return results

# Entry point
if __name__ == "__main__":
    csv_file = "your_dataset.csv"
    optimized_model, optimized_prompt = optimize_phi4_prompts(csv_file, save_prompt=True)

    if optimized_model:
        test_queries = [
            "I hate all immigrants and they should be deported.",
            "The weather is nice today and I'm enjoying the sunshine.",
            "Women should stay at home and not be allowed to work."
        ]
        test_optimized_model(optimized_model, test_queries, optimized_prompt)
