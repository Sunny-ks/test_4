import dspy
from dspy.teleprompt import BootstrapFewShot
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load local Phi-4 model
model_path = './phi4'  # Change to your actual local path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# 2. Wrap in HuggingFace pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False,  # deterministic output
    return_full_text=False
)

# 3. Set the DSPy LLM
dspy.settings.configure(
    lm=dspy.HFTextGenerationLM(pipeline=text_gen_pipeline)
)

# 4. Define DSPy Signature
class ExplainConcept(dspy.Signature):
    """Explain a concept simply."""
    concept = dspy.InputField(desc="The concept to explain")
    explanation = dspy.OutputField(desc="Explanation in simple terms")

# 5. Define DSPy Module
class Explainer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.explainer = dspy.Predict(ExplainConcept)

    def forward(self, concept):
        return self.explainer(concept=concept)

# 6. Few-shot training examples
train_set = [
    dspy.Example(concept="machine learning", explanation="It's like teaching computers to learn from data.").with_inputs("concept"),
    dspy.Example(concept="neural networks", explanation="They're computer models inspired by the brain to recognize patterns.").with_inputs("concept"),
]

# 7. Compile with BootstrapFewShot
compiler = BootstrapFewShot(metric=dspy.evaluate.match_answer)
compiled_explainer = compiler.compile(Explainer(), train_set=train_set)

# 8. Inference
output = compiled_explainer(concept="large language model")
print("Explanation:", output.explanation)
