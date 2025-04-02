from typing import Dict, List, Any, Callable, Optional, Union
import asyncio
from abc import ABC, abstractmethod

class ModelResponse:
    """Standardized model response class"""
    def __init__(self, id: str, choices: List[Dict], model: str, usage: Optional[Dict] = None):
        self.id = id
        self.choices = [Choice(**choice) for choice in choices]
        self.model = model
        self.usage = usage

class Choice:
    """Represents a single completion choice"""
    def __init__(self, index: int, message: Dict, finish_reason: Optional[str] = None):
        self.index = index
        self.message = Message(**message)
        self.finish_reason = finish_reason

class Message:
    """Represents a message in a choice"""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class ModelProvider(ABC):
    """Base interface for language model providers"""
    
    @abstractmethod
    def completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> ModelResponse:
        """Synchronous completion request to the provider.
        
        Args:
            model: The model identifier to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse object containing the model's output
        """
        pass
    
    async def async_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> ModelResponse:
        """Asynchronous completion request to the provider.
        
        By default, this creates a wrapper around the synchronous completion method.
        Providers should override this with a native async implementation when possible.
        
        Args:
            model: The model identifier to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse object containing the model's output
        """
        # Default implementation runs the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.completion(model, messages, temperature, max_tokens, **kwargs)
        )
    
    @abstractmethod
    def format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert a list of messages to a single prompt string format expected by the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def streaming_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048, callback: Callable[[str], None] = None, **kwargs) -> ModelResponse:
        """Stream tokens as they are generated.
        
        This is an optional method that providers can implement for streaming support.
        
        Args:
            model: The model identifier to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            callback: Function to call with each generated token
            **kwargs: Additional model-specific parameters
            
        Returns:
            Final ModelResponse object containing the complete model's output
        """
        # Default implementation just calls regular completion and doesn't stream
        response = self.completion(model, messages, temperature, max_tokens, **kwargs)
        if callback and response.choices:
            callback(response.choices[0].message.content)
        return response
    
    async def async_streaming_completion(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048, callback: Callable[[str], None] = None, **kwargs) -> ModelResponse:
        """Asynchronously stream tokens as they are generated.
        
        This is an optional method that providers can implement for async streaming support.
        
        Args:
            model: The model identifier to use
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            callback: Function to call with each generated token
            **kwargs: Additional model-specific parameters
            
        Returns:
            Final ModelResponse object containing the complete model's output
        """
        # Default implementation calls async_completion and doesn't stream
        response = await self.async_completion(model, messages, temperature, max_tokens, **kwargs)
        if callback and response.choices:
            callback(response.choices[0].message.content)
        return response