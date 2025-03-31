import pandas as pd
import dspy
from dspy.teleprompt import Teleprompter, BootstrapFewShot
from dspy.evaluate import Evaluate
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- STEP 0: OpenAI as prompt teacher ---------- #
def gpt4_generate_prompts(example_texts, task_description):
    client = OpenAI(api_key="sk-...")  # Replace with your OpenAI key
    prompt = f"""
You are a prompt engineer specializing in optimizing prompts for Microsoft's Phi-4 model.
Phi-4 is a smaller language model (4B parameters) that performs best with clear, concise prompts.

IMPORTANT: Phi-4 uses a specific tagging format:
<|im_start|>system<|im_sep|>
[System message here]<|im_end|>
<|im_start|>user<|im_sep|>
[User message here]<|im_end|>
<|im_start|>assistant<|im_sep|>
[Assistant response here]<|im_end|>

TASK DESCRIPTION:
{task_description}

EXAMPLES:
{example_texts}

Generate 5 improved prompt templates specifically optimized for Phi-4 to classify such queries.
Each prompt should:
1. Use the proper Phi-4 tagging format (<|im_start|>system<|im_sep|>, etc.)
2. Include clear classification instructions in the system message
3. Format the input query as part of the user message
4. Be concise and direct (Phi-4 has limited context window)
5. Utilize any patterns you observe in the examples

For each prompt, include a brief explanation of why it might work well with Phi-4.
"""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    outputs = response.choices[0].message.content
    # Parse the outputs to extract just the prompts
    prompts = []
    for line in outputs.split("\n"):
        if line.startswith("Prompt") and ":" in line:
            prompts.append(line.split(":", 1)[1].strip())
    return prompts if prompts else outputs.strip().split("\n\n")

# ---------- STEP 1: Load and prepare dataset ---------- #
def prepare_dataset(csv_path, text_col="text", label_col="label", train_ratio=0.8):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    
    # Get label distribution for better few-shot examples
    label_counts = df[label_col].value_counts()
    print(f"Label distribution:\n{label_counts}")
    
    # Create DSPy examples
    examples = [dspy.Example(input_text=row[text_col], label=row[label_col]) 
                for _, row in df.iterrows()]
    
    # Stratified split based on labels
    train_examples = []
    dev_examples = []
    
    # Group examples by label
    by_label = {}
    for ex in examples:
        if ex.label not in by_label:
            by_label[ex.label] = []
        by_label[ex.label].append(ex)
    
    # Split each label group according to train_ratio
    for label, label_examples in by_label.items():
        split = int(train_ratio * len(label_examples))
        train_examples.extend(label_examples[:split])
        dev_examples.extend(label_examples[split:])
    
    # Shuffle examples
    np.random.shuffle(train_examples)
    np.random.shuffle(dev_examples)
    
    return train_examples, dev_examples

# ---------- STEP 2: Configure Phi-4 model ---------- #
def setup_phi4_model(model_name_or_path, device="cuda"):
    """Load and configure Phi-4 model with DSPy"""
    # Check if using a Hugging Face model or local path
    is_local = "/" in model_name_or_path
    
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    
    # Load directly with transformers for more control
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    
    # Configure phi4 model with DSPy
    phi4_lm = dspy.HFModel(
        model=model,
        tokenizer=tokenizer,
        max_tokens=512,
        temperature=0.1,  # Keep temperature low for classification
        device=device
    )
    
    return phi4_lm

# ---------- STEP 3: Advanced prompt optimization with DSPy ---------- #
def optimize_prompts(trainset, devset, phi4_lm, suggested_prompts, task_description):
    dspy.settings.configure(lm=phi4_lm)
    
    # Track metrics for all prompts
    results = []
    
    # Try all GPT-4 generated prompts
    for i, prompt_text in enumerate(suggested_prompts):
        print(f"\n🔍 Trying Prompt #{i+1}:\n{prompt_text}\n")
        
        class PromptedClassifier(dspy.Program):
            def __init__(self):
                # Format the prompt with Phi-4's expected tagging structure
                phi4_formatted_prompt = f"""<|im_start|>system<|im_sep|>
{prompt_text}<|im_end|>
<|im_start|>user<|im_sep|>
{{input_text}}<|im_end|>
<|im_start|>assistant<|im_sep|>"""
                
                self.classify = dspy.Predict(
                    "input_text -> label",
                    prompt=phi4_formatted_prompt
                )
            
            def forward(self, input_text):
                return self.classify(input_text=input_text)
        
        # Basic prompt testing
        evaluator = Evaluate(devset=devset[:50], metric="accuracy")  # Use subset for speed
        basic_program = PromptedClassifier()
        basic_score = evaluator(basic_program)
        print(f"Base Accuracy: {basic_score * 100:.2f}%")
        
        # Try with Teleprompter optimization
        teleprompter = Teleprompter(PromptedClassifier(), metric="accuracy")
        compiled_program = teleprompter.compile(trainset=trainset[:50])  # Use subset for speed
        teleprompter_score = evaluator(compiled_program)
        print(f"Teleprompter Accuracy: {teleprompter_score * 100:.2f}%")
        
        # Try with bootstrapped few-shot examples
        bootstrapper = BootstrapFewShot(PromptedClassifier(), metric="accuracy")
        bootstrapped_program = bootstrapper.compile(trainset=trainset[:100])
        bootstrap_score = evaluator(bootstrapped_program)
        print(f"BootstrapFewShot Accuracy: {bootstrap_score * 100:.2f}%")
        
        # Store results
        results.append({
            "prompt": prompt_text,
            "basic_score": basic_score,
            "teleprompter_score": teleprompter_score,
            "bootstrap_score": bootstrap_score,
            "compiled_teleprompter": compiled_program,
            "compiled_bootstrap": bootstrapped_program,
            "max_score": max(basic_score, teleprompter_score, bootstrap_score)
        })
    
    # Find the best performer
    best_result = max(results, key=lambda x: x["max_score"])
    best_score_type = max(
        ["basic", "teleprompter", "bootstrap"], 
        key=lambda t: best_result[f"{t}_score"]
    )
    best_program = best_result[f"compiled_{best_score_type}"] if best_score_type != "basic" else PromptedClassifier()
    
    # Get the compiled prompt for inspection
    def get_compiled_prompt(program):
        try:
            # Use a representative example for better prompt visualization
            example_input = "Example query to classify"
            compiled_prompt = program.lm.compile_prompt(
                "input_text -> label",
                input_text=example_input  # dummy input to expand prompt
            )
            
            # Clean up the compiled prompt for better readability
            # This ensures we can see the actual Phi-4 format with im_start tags
            cleaned_prompt = compiled_prompt.replace("\\n", "\n")
            return cleaned_prompt
        except Exception as e:
            print(f"Error extracting compiled prompt: {e}")
            return "Couldn't extract compiled prompt"
    
    best_compiled_prompt = get_compiled_prompt(best_program)
    
    print("\n🏆 Best Result:")
    print(f"Prompt: {best_result['prompt']}")
    print(f"Method: {best_score_type}")
    print(f"Accuracy: {best_result['max_score'] * 100:.2f}%")
    
    return best_program, best_result, best_compiled_prompt

# ---------- STEP 4: Comprehensive evaluation ---------- #
def evaluate_best_model(best_program, devset):
    true_labels = []
    pred_labels = []
    
    for example in devset:
        true_labels.append(example.label)
        result = best_program(input_text=example.input_text)
        pred_labels.append(result.label)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    
    print("\n📊 Final Evaluation:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, report, (true_labels, pred_labels)

# ---------- STEP 5: Save the optimized prompt for reuse ---------- #
def save_optimized_prompt(best_prompt, compiled_prompt, metrics, output_file="optimized_phi4_prompt.json"):
    import json
    import datetime
    
    prompt_data = {
        "date_generated": datetime.datetime.now().isoformat(),
        "base_prompt": best_prompt,
        "compiled_prompt": compiled_prompt,
        "accuracy": metrics[0],
        "performance": {
            "accuracy": metrics[0],
            "report": str(metrics[1])
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(prompt_data, f, indent=2)
    
    print(f"\n💾 Saved optimized prompt to {output_file}")

# ---------- Main Execution ---------- #
def main():
    # Configuration
    csv_path = "your_dataset.csv"  # Replace with your file path
    task_description = "Classify user queries about electric vehicles into intent categories"
    phi4_model_path = "your-finetuned-phi4"  # Replace with your model path
    
    # Step 1: Prepare dataset
    trainset, devset = prepare_dataset(csv_path)
    print(f"Dataset loaded: {len(trainset)} training examples, {len(devset)} validation examples")
    
    # Step 2: Setup Phi-4 model
    phi4_lm = setup_phi4_model(phi4_model_path)
    
    # Step 3: Generate prompt suggestions
    example_texts = "\n".join([f"- \"{ex.input_text}\" → {ex.label}" 
                              for ex in trainset[:10]])  # Use 10 examples
    suggested_prompts = gpt4_generate_prompts(example_texts, task_description)
    
    # Step 4: Optimize prompts
    best_program, best_result, best_compiled_prompt = optimize_prompts(
        trainset, devset, phi4_lm, suggested_prompts, task_description
    )
    
    # Step 5: Evaluate
    metrics = evaluate_best_model(best_program, devset)
    
    # Step 6: Save optimized prompt
    save_optimized_prompt(best_result["prompt"], best_compiled_prompt, metrics)
    
    # Step 7: Test on new inputs
    test_inputs = [
        "How do I preheat my EV?",
        "Where are charging stations in Boston?",
        "What's the range on a full charge?"
    ]
    
    print("\n🧪 Testing with new inputs:")
    for input_text in test_inputs:
        result = best_program(input_text=input_text)
        print(f"Input: {input_text}")
        print(f"Predicted Intent: {result.label}\n")

if __name__ == "__main__":
    main()
