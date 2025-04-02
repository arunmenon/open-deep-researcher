from typing import Dict, List, Any, Tuple

from ..models.provider import ModelProvider

class SearchService:
    """Service for performing search operations"""
    
    def __init__(self, model_provider: ModelProvider, model_name: str):
        """Initialize the search service
        
        Args:
            model_provider: The model provider to use
            model_name: The name of the model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
        # Add a simple in-memory cache
        self.cache = {}
    
    async def search(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Perform a search using the model provider
        
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
            # Create system and user messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides comprehensive research information on topics. Please provide detailed and accurate information about the following query, including relevant facts, figures, dates, and analysis."},
                {"role": "user", "content": f"Please research and provide detailed information about: {query}"}
            ]
            
            # Make the API call through our model provider
            print(f"Performing search for query: {query}")
            
            response = await self.model_provider.async_completion(
                model=self.model_name,
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