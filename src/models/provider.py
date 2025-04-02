from typing import Dict, List, Any
from abc import ABC, abstractmethod

class ModelProvider(ABC):
    """Base interface for language model providers"""
    
    @abstractmethod
    def completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048, **kwargs):
        """
        Make a completion request to the provider.
        
        Args:
            model: The model identifier to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Response object containing the model's output
        """
        pass
    
    @abstractmethod
    def format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of messages to a single prompt string format expected by the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        pass