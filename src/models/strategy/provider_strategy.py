"""
Strategy implementation that wraps existing ModelProvider instances.
"""

from typing import Dict, List, Any

from ..provider import ModelProvider, ModelResponse
from .base_strategy import ModelStrategy


class ProviderModelStrategy(ModelStrategy):
    """
    Strategy implementation that wraps a ModelProvider
    
    This allows existing provider implementations to be used with
    the new strategy pattern.
    """
    
    def __init__(self, provider: ModelProvider, model_name: str):
        """
        Initialize with a provider and model name
        
        Args:
            provider: The model provider instance
            model_name: The name of the model to use
        """
        self.provider = provider
        self.model_name = model_name
        
    async def execute_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """Execute completion via the provider"""
        return await self.provider.async_completion(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings via the provider
        
        Note: Not all providers support embeddings.
        """
        if hasattr(self.provider, 'get_embeddings'):
            return await self.provider.get_embeddings(text)
        # Return empty list if embeddings not supported
        return []
        
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Return capabilities based on provider"""
        # Determine capabilities based on provider method availability
        return {
            "streaming": hasattr(self.provider, 'async_streaming_completion'),
            "embeddings": hasattr(self.provider, 'get_embeddings'),
            "function_calling": False  # Add implementation if needed
        }