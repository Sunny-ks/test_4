import dspy

# Specify the model name from the Hugging Face Hub
model_name = "your-fine-tuned-model-name"  # Replace with your model's name

# Initialize the language model with the specified provider
lm = dspy.LM(model=model_name, provider="huggingface", device_map="auto")

# Configure DSPy to use the initialized language model
dspy.configure(lm=lm)# Define a simple question-answering module


class SimpleQA(dspy.Signature):
    """A simple question-answering model."""
    question = dspy.InputField()
    answer = dspy.OutputField()

qa_module = dspy.ChainOfThought(SimpleQA)

# Test the module with a sample question
response = qa_module(question="What is the capital of France?")
print(response.answer)



