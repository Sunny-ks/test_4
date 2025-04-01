import os
import torch
import time
import logging
import openai
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API setup - make sure to set your API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

class GPT4oRewardModel:
    def __init__(self, cache_dir: Optional[str] = None):
        self.client = openai.OpenAI()
        self.system_prompt = """
        You are evaluating the quality of a response to a given system prompt and user query.
        
        Rate the response on a scale from 1 to 10 based on the following criteria:
        
        1 = UNACCEPTABLE: Response is completely incorrect, potentially harmful, irrelevant to the query, or violates ethical guidelines.
        
        2 = VERY POOR: Response is mostly incorrect, off-topic, or unhelpful. It fails to address the main points of the query.
        
        3 = POOR: Response attempts to address the query but has major gaps, inaccuracies, or misunderstandings.
        
        4 = BELOW AVERAGE: Response partially addresses the query but lacks depth, contains several errors, or is disorganized.
        
        5 = AVERAGE: Response is factually adequate but basic. It addresses the main query but lacks nuance or detail.
        
        6 = ABOVE AVERAGE: Response is mostly accurate and provides useful information. It addresses the main query well.
        
        7 = GOOD: Response is accurate, clear, and addresses all aspects of the query with minimal errors.
        
        8 = VERY GOOD: Response is comprehensive, well-structured, accurate, and tailored to the query.
        
        9 = EXCELLENT: Response is exceptionally thorough, accurate, insightful, and well-communicated.
        
        10 = OUTSTANDING: Response is perfect - comprehensive, precise, nuanced, and provides exceptional value.
        
        Provide only a numerical score (1-10) without any explanation.
        """
        
        # Setup caching
        self.cache_dir = cache_dir
        self.cache = {}
        if cache_dir and os.path.exists(f"{cache_dir}/reward_cache.pt"):
            logger.info("Loading reward cache from disk...")
            self.cache = torch.load(f"{cache_dir}/reward_cache.pt")
            logger.info(f"Loaded {len(self.cache)} cached rewards")
    
    def get_reward(self, system_prompt: str, user_query: str, response: str) -> float:
        # Create a cache key from the inputs
        cache_key = f"{hash(system_prompt)}-{hash(user_query)}-{hash(response)}"
        
        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"System prompt: {system_prompt}\n\nUser query: {user_query}\n\nModel response: {response}"}
            ]
            
            # Add exponential backoff for API rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.0,
                        max_tokens=10
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"API error: {e}. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
            
            # Extract numerical score from response
            score_text = completion.choices[0].message.content.strip()
            logger.info(f"Raw score from GPT-4o: {score_text}")
            
            try:
                score = float(score_text)
                # Normalize to range [-1, 1] for RLHF
                normalized_score = (score - 5.5) / 4.5
                
                # Cache the result
                if self.cache_dir:
                    self.cache[cache_key] = normalized_score
                    # Save cache every time in test mode
                    os.makedirs(self.cache_dir, exist_ok=True)
                    torch.save(self.cache, f"{self.cache_dir}/reward_cache.pt")
                        
                return normalized_score
            except ValueError:
                logger.warning(f"Failed to parse score from: {score_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting reward: {e}")
            return 0.0
    
    # New method that matches what GRPOTrainer expects
    def batch_evaluate(self, prompts, completions, **kwargs) -> List[float]:
        """
        Process prompts and completions as provided by GRPOTrainer and compute rewards.
        """
        results = []
        
        logger.info(f"batch_evaluate received {len(prompts)} prompts and {len(completions)} completions")
        logger.info(f"Prompt example type: {type(prompts[0])}")
        if isinstance(prompts[0], dict):
            logger.info(f"Prompt example keys: {prompts[0].keys()}")
        logger.info(f"Completion example type: {type(completions[0])}")
            
        for prompt, completion in zip(prompts, completions):
            # Extract system prompt and user query based on the dataset format
            if isinstance(prompt, dict) and 'system_prompt' in prompt and 'user_query' in prompt:
                # Dictionary format
                system_prompt = prompt['system_prompt']
                user_query = prompt['user_query']
                logger.info(f"Extracted from dict format - System: {system_prompt[:30]}... User: {user_query[:30]}...")
            elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
                # Conversational format
                system_prompt = next((msg["content"] for msg in prompt if msg["role"] == "system"), "")
                user_query = next((msg["content"] for msg in prompt if msg["role"] == "user"), "")
                logger.info(f"Extracted from conversational format - System: {system_prompt[:30]}... User: {user_query[:30]}...")
            else:
                # Plain text format - assuming system prompt and user query are separated by newlines
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                parts = prompt_str.split("\n\n")
                system_prompt = parts[0] if len(parts) > 1 else ""
                user_query = parts[-1]
                logger.info(f"Extracted from text format - System: {system_prompt[:30]}... User: {user_query[:30]}...")
            
            # Extract the completion/response
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                # Conversational format response
                response = completion[0].get("content", "")
                logger.info(f"Extracted response from conversational format: {response[:30]}...")
            else:
                # Plain text response
                response = completion if isinstance(completion, str) else str(completion)
                logger.info(f"Extracted response from text format: {response[:30]}...")
            
            # Get reward for this pair
            reward = self.get_reward(system_prompt, user_query, response)
            logger.info(f"Calculated reward: {reward}")
            results.append(reward)
            
        return results


def test_individual_reward():
    """Test individual get_reward method"""
    model = GPT4oRewardModel(cache_dir="./test_reward_cache")
    
    system_prompt = "You are a helpful AI assistant that provides accurate and concise information."
    user_query = "What are the main causes of climate change?"
    
    # Test with a good response
    good_response = """Climate change is primarily caused by the greenhouse effect, where certain gases trap heat in Earth's atmosphere. The main causes include:

1. Burning fossil fuels like coal, oil, and natural gas, which releases carbon dioxide
2. Deforestation, which reduces the planet's ability to absorb carbon dioxide
3. Industrial processes that release methane and other greenhouse gases
4. Agricultural practices, particularly livestock farming
5. Increasing global population and consumption patterns

Human activities have accelerated these processes significantly since the industrial revolution, leading to rising global temperatures, changing weather patterns, and other climate impacts."""

    # Test with a poor response
    poor_response = "Climate change is caused by stuff like pollution and things that happen naturally."
    
    # Get rewards
    good_reward = model.get_reward(system_prompt, user_query, good_response)
    poor_reward = model.get_reward(system_prompt, user_query, poor_response)
    
    print(f"\nIndividual Reward Test Results:")
    print(f"Good response reward: {good_reward} (normalized from 1-10 scale)")
    print(f"Poor response reward: {poor_reward} (normalized from 1-10 scale)")


def test_batch_evaluate():
    """Test batch_evaluate method with different input formats"""
    model = GPT4oRewardModel(cache_dir="./test_reward_cache")
    
    # Test with dictionary format (what we'll use with GRPO)
    dict_prompts = [
        {"system_prompt": "You are a helpful assistant.", "user_query": "What is machine learning?"},
        {"system_prompt": "You are a math tutor.", "user_query": "Explain calculus in simple terms."}
    ]
    
    dict_completions = [
        "Machine learning is a branch of artificial intelligence that uses data to learn patterns and make predictions without explicit programming.",
        "Calculus is a branch of mathematics that studies continuous change. It has two main branches: differential calculus and integral calculus."
    ]
    
    # Test with conversational format
    conv_prompts = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ],
        [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Explain calculus in simple terms."}
        ]
    ]
    
    conv_completions = [
        [{"role": "assistant", "content": "Machine learning is a branch of artificial intelligence that uses data to learn patterns and make predictions without explicit programming."}],
        [{"role": "assistant", "content": "Calculus is a branch of mathematics that studies continuous change. It has two main branches: differential calculus and integral calculus."}]
    ]
    
    # Test with plain text format
    text_prompts = [
        "You are a helpful assistant.\n\nWhat is machine learning?",
        "You are a math tutor.\n\nExplain calculus in simple terms."
    ]
    
    text_completions = [
        "Machine learning is a branch of artificial intelligence that uses data to learn patterns and make predictions without explicit programming.",
        "Calculus is a branch of mathematics that studies continuous change. It has two main branches: differential calculus and integral calculus."
    ]
    
    # Get rewards for each format
    print("\nBatch Evaluate Test Results:")
    
    print("\n1. Dictionary Format Test:")
    dict_rewards = model.batch_evaluate(dict_prompts, dict_completions)
    for i, reward in enumerate(dict_rewards):
        print(f"Example {i+1} reward: {reward}")
    
    print("\n2. Conversational Format Test:")
    conv_rewards = model.batch_evaluate(conv_prompts, conv_completions)
    for i, reward in enumerate(conv_rewards):
        print(f"Example {i+1} reward: {reward}")
    
    print("\n3. Plain Text Format Test:")
    text_rewards = model.batch_evaluate(text_prompts, text_completions)
    for i, reward in enumerate(text_rewards):
        print(f"Example {i+1} reward: {reward}")


if __name__ == "__main__":
    print("=== Testing GPT-4o Reward Model ===")
    
    # First test the individual reward function
    test_individual_reward()
    
    # Then test the batch_evaluate method with different input formats
    test_batch_evaluate()
    
    print("\nAll tests completed!")
