import dspy

# === 1. Initialization ===
# Set your vLLM server model endpoint details
model_name = "hosted_vllm/deepseek-ai/deepseek-llm-7b-chat"  # Can be anything, but used for tracking
api_base_url = "http://localhost:8000/v1/"  # Must match your vLLM server

# Initialize the DSPy-compatible LM client using vLLM
vllm_model = dspy.LM(
    model=model_name,
    api_base=api_base_url,
    provider=dspy.LocalProvider(),  # Local/OpenAI-compatible API (like vLLM)
    temperature=0.7,
    max_tokens=100,
    cache=False  # Set to True if you want caching for repeated queries
)

# Set this model as the global default for DSPy
dspy.configure(lm=vllm_model)

# === 2. Define a simple QA task ===
qa_module = dspy.ChainOfThought("question -> answer")

# === 3. Run a test question ===
response = qa_module(question="What is the capital of France?")

# === 4. Show result ===
print("Response:", response.answer)
