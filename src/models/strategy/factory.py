"""
Factory for creating model strategy instances.
"""

from typing import Dict, Any

from .base_strategy import ModelStrategy
from .runpod_strategy import RunPodStrategy


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