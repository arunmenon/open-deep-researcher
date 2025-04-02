from typing import Dict, List, Any, Tuple

from ..models.strategy import ModelStrategy
from ..prompts import system_prompts, user_prompts

class SearchService:
    """Service for performing search operations"""
    
    def __init__(self, model_strategy: ModelStrategy):
        """Initialize the search service
        
        Args:
            model_strategy: The model strategy to use
        """
        self.model_strategy = model_strategy
        # Add a simple in-memory cache
        self.cache = {}
    
    async def search(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Perform a search using the model strategy
        
        Args:
            query: The search query
            
        Returns:
            Tuple of (result_text, sources_dict)
        """
        # Check cache first
        if query in self.cache:
            print(f"Using cached result for query: {query}")
            return self.cache[query]
            
        try:
            # Get the prompt from the externalized prompts module
            user_prompt = user_prompts.search_prompt(query)
            
            # Create system and user messages
            messages = [
                {"role": "system", "content": system_prompts.SEARCH_ASSISTANT},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make the API call through our model strategy
            print(f"Performing search for query: {query}")
            
            response = await self.model_strategy.execute_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=4096
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Create empty sources since we're not using a search tool
            sources = {}
            
            # Cache the result
            result = (response_text, sources)
            self.cache[query] = result
            
            return result
            
        except Exception as e:
            print(f"Error performing search: {e}")
            # Return empty results in case of failure
            return f"Error performing search: {e}", {}