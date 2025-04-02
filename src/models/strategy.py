"""
Model strategy pattern implementation.

This module defines the strategy interface and concrete implementations
for different model providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable

from .provider import ModelProvider, ModelResponse


class ModelStrategy(ABC):
    """
    Strategy interface for different model implementations.
    
    This abstraction allows for different model backends to be used
    interchangeably without affecting the core research logic.
    """
    
    @abstractmethod
    async def execute_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """
        Execute a completion request against the model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters specific to the model
            
        Returns:
            ModelResponse containing the completion result
        """
        pass
        
    @abstractmethod
    async def get_embeddings(self, text: str) -> List[float]:
        """
        Get vector embeddings for text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        pass
        
    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, bool]:
        """
        Return model capabilities
        
        Returns:
            Dictionary of capability flags
        """
        pass


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


class RunPodStrategy(ProviderModelStrategy):
    """
    Strategy implementation specifically for RunPod
    
    This provides RunPod-specific configuration and defaults.
    """
    
    def __init__(self, api_key: str, endpoint_id: str, model_name: str = "runpod/mistral-24b-instruct"):
        """
        Initialize RunPod strategy
        
        Args:
            api_key: RunPod API key
            endpoint_id: RunPod endpoint ID
            model_name: Model name (defaults to Mistral)
        """
        from .runpod_provider import RunpodProvider
        provider = RunpodProvider(api_key, endpoint_id)
        super().__init__(provider, model_name)
        
    @property
    def capabilities(self) -> Dict[str, bool]:
        """RunPod-specific capabilities"""
        capabilities = super().capabilities
        # Add any RunPod-specific capability overrides
        return capabilities


class ModelStrategyFactory:
    """
    Factory for creating model strategy instances
    
    This centralizes strategy creation logic and configuration.
    """
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any]) -> ModelStrategy:
        """
        Create a strategy based on type and configuration
        
        Args:
            strategy_type: The type of strategy ('runpod', etc.)
            config: Configuration parameters for the strategy
            
        Returns:
            ModelStrategy instance
            
        Raises:
            ValueError: If strategy_type is unknown
        """
        if strategy_type == "runpod":
            return RunPodStrategy(
                config.get("api_key"),
                config.get("endpoint_id"),
                config.get("model_name", "runpod/mistral-24b-instruct")
            )
        # Add more strategy types as needed
        raise ValueError(f"Unknown strategy type: {strategy_type}")