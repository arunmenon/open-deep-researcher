"""
Base strategy interface for model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional

from ..provider import ModelResponse


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