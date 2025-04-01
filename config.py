class GPT4oRewardModel:
    def __init__(self, cache_dir: Optional[str] = None):
        self.client = openai.OpenAI()
        self.system_prompt = """
        You are evaluating the quality of a response to a given system prompt and user query.
        
        Rate the response on a scale from 1 to 10 based on the following criteria:
        
        1 = UNACCEPTABLE: Response is completely incorrect, potentially harmful, irrelevant to the query, or violates ethical guidelines. The response may contain dangerous misinformation or inappropriate content.
        
        2 = VERY POOR: Response is mostly incorrect, off-topic, or unhelpful. It fails to address the main points of the query and may contain significant errors.
        
        3 = POOR: Response attempts to address the query but has major gaps, inaccuracies, or misunderstandings. It provides little value to the user.
        
        4 = BELOW AVERAGE: Response partially addresses the query but lacks depth, contains several errors, or is disorganized. The information provided is limited in usefulness.
        
        5 = AVERAGE: Response is factually adequate but basic. It addresses the main query but lacks nuance, detail, or may miss secondary aspects of the query.
        
        6 = ABOVE AVERAGE: Response is mostly accurate and provides useful information. It addresses the main query well but may miss minor details or opportunities to be more helpful.
        
        7 = GOOD: Response is accurate, clear, and addresses all aspects of the query. It provides valuable information with minimal errors or omissions.
        
        8 = VERY GOOD: Response is comprehensive, well-structured, accurate, and tailored to the query. It shows good understanding of context and provides insightful information.
        
        9 = EXCELLENT: Response is exceptionally thorough, accurate, insightful, and well-communicated. It anticipates related questions and provides additional helpful context.
        
        10 = OUTSTANDING: Response is perfect in every way - comprehensive, precise, nuanced, tailored perfectly to the query, and provides exceptional value. It represents the ideal response that could not be reasonably improved.
        
        Provide only a numerical score (1-10) without any explanation.
        """
        
        # Setup caching (optional)
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
            max_retries = 5
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
            try:
                score = float(score_text)
                # Normalize to range [-1, 1] for RLHF
                normalized_score = (score - 5.5) / 4.5
                
                # Cache the result
                if self.cache_dir:
                    self.cache[cache_key] = normalized_score
                    if len(self.cache) % 100 == 0:  # Save cache periodically
                        os.makedirs(self.cache_dir, exist_ok=True)
                        torch.save(self.cache, f"{self.cache_dir}/reward_cache.pt")
                        
                return normalized_score
            except ValueError:
                logger.warning(f"Failed to parse score from: {score_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting reward: {e}")
            return 0.0
    
    # This is the updated method that matches what GRPOTrainer expects
    def batch_evaluate(self, prompts, completions, **kwargs) -> List[float]:
        """
        Process prompts and completions as provided by GRPOTrainer and compute rewards.
        
        Args:
            prompts: List of prompts (raw data from dataset)
            completions: List of generated completions
            **kwargs: Additional arguments passed by GRPOTrainer
            
        Returns:
            List of reward scores normalized to [-1, 1]
        """
        results = []
        
        # Debug information
        if len(prompts) > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Prompt type: {type(prompts[0])}")
            logger.debug(f"Completion type: {type(completions[0])}")
            
        for prompt, completion in zip(prompts, completions):
            # Extract system prompt and user query based on the dataset format
            if isinstance(prompt, dict) and 'system_prompt' in prompt and 'user_query' in prompt:
                # Direct dictionary format
                system_prompt = prompt['system_prompt']
                user_query = prompt['user_query']
            elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
                # Conversational format
                system_prompt = next((msg["content"] for msg in prompt if msg["role"] == "system"), "")
                user_query = next((msg["content"] for msg in prompt if msg["role"] == "user"), "")
            else:
                # Plain text format - assuming system prompt and user query are separated by newlines
                # Adjust this parsing logic based on your actual data format
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                parts = prompt_str.split("\n\n")
                system_prompt = parts[0] if len(parts) > 1 else ""
                user_query = parts[-1]
            
            # Extract the completion/response
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                # Conversational format response
                response = completion[0].get("content", "")
            else:
                # Plain text response
                response = completion if isinstance(completion, str) else str(completion)
            
            # Get reward for this pair
            reward = self.get_reward(system_prompt, user_query, response)
            results.append(reward)
            
        return results
