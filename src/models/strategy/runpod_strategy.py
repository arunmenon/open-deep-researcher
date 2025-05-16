"""
Strategy implementation for RunPod.
"""

from typing import Dict, List, Any

from .provider_strategy import ProviderModelStrategy


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
        from ..runpod_provider import RunpodProvider
        provider = RunpodProvider(api_key, endpoint_id)
        super().__init__(provider, model_name)
        
    @property
    def capabilities(self) -> Dict[str, bool]:
        """RunPod-specific capabilities"""
        capabilities = super().capabilities
        # Add any RunPod-specific capability overrides
        return capabilities