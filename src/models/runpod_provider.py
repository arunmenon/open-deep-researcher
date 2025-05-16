import datetime
import requests
import json
import time
from typing import Dict, List, Any, Callable, Optional, Union
import asyncio
import aiohttp

from .provider import ModelProvider, ModelResponse, Choice, Message

class RunpodProvider(ModelProvider):
    """Provider for RunPod API."""
    
    def __init__(self, api_key, endpoint_id):
        """Initialize RunPod provider with API key and endpoint ID"""
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.api_base = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.status_base = f"https://api.runpod.ai/v2/{endpoint_id}/status"
        # Add LRU cache to store recent responses
        self.response_cache = {}
        self.max_cache_size = 100
        self.request_semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
    def completion(self, model, messages, temperature=0.7, max_tokens=2048, **kwargs):
        """Make a synchronous completion request to RunPod API."""
        # Format the messages for Mistral
        formatted_prompt = self.format_messages_to_prompt(messages)
        
        # Generate cache key
        cache_key = self._generate_cache_key(formatted_prompt, temperature, max_tokens, kwargs)
        
        # Check cache
        if cache_key in self.response_cache:
            print("Using cached response")
            return self.response_cache[cache_key]
        
        # Prepare the payload
        payload = {
            "input": {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.95)
            }
        }
        
        # Headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Make the API call
        print(f"Making API call to RunPod endpoint: {self.endpoint_id}")
        
        response = requests.post(
            self.api_base,
            headers=headers,
            json=payload
        )
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"RunPod API error: {response.status_code}, {response.text}")
        
        # Get the job ID and poll for results
        result = response.json()
        job_id = result.get("id")
        
        if not job_id:
            raise Exception(f"No job ID returned from RunPod API. Response: {result}")
        
        print(f"RunPod job submitted with ID: {job_id}. Polling for results...")
        
        # Poll for results with exponential backoff
        response_data = self._poll_for_results(job_id, headers)
        
        # Cache response
        self._add_to_cache(cache_key, response_data)
        
        return response_data
    
    async def async_completion(self, model, messages, temperature=0.7, max_tokens=2048, **kwargs):
        """Make an asynchronous completion request to RunPod API."""
        # Format the messages for Mistral
        formatted_prompt = self.format_messages_to_prompt(messages)
        
        # Generate cache key
        cache_key = self._generate_cache_key(formatted_prompt, temperature, max_tokens, kwargs)
        
        # Check cache
        if cache_key in self.response_cache:
            print("Using cached response")
            return self.response_cache[cache_key]
        
        # Prepare the payload
        payload = {
            "input": {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.95)
            }
        }
        
        # Headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Use a semaphore to limit concurrent requests
        async with self.request_semaphore:
            # Make the API call
            print(f"Making async API call to RunPod endpoint: {self.endpoint_id}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_base,
                    headers=headers,
                    json=payload
                ) as response:
                    # Check for errors
                    if response.status != 200:
                        text = await response.text()
                        raise Exception(f"RunPod API error: {response.status}, {text}")
                    
                    # Get the job ID and poll for results
                    result = await response.json()
                    job_id = result.get("id")
                    
                    if not job_id:
                        raise Exception(f"No job ID returned from RunPod API. Response: {result}")
                    
                    print(f"RunPod job submitted with ID: {job_id}. Polling for results...")
                    
                    # Poll for results with exponential backoff
                    response_data = await self._async_poll_for_results(job_id, headers)
                    
                    # Cache response
                    self._add_to_cache(cache_key, response_data)
                    
                    return response_data
    
    def _poll_for_results(self, job_id, headers):
        """Poll for results with exponential backoff (synchronous)"""
        max_attempts = 10
        wait_time = 2  # Start with 2 seconds
        
        for attempt in range(max_attempts):
            print(f"Polling attempt {attempt+1}/{max_attempts}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            
            # Check job status
            status_url = f"{self.status_base}/{job_id}"
            status_response = requests.get(
                status_url,
                headers=headers
            )
            
            if status_response.status_code != 200:
                print(f"Error checking job status: {status_response.status_code}, {status_response.text}")
                wait_time = min(wait_time * 2, 30)  # Exponential backoff, max 30 seconds
                continue
            
            status_result = status_response.json()
            status = status_result.get("status")
            
            print(f"Job status: {status}")
            
            if status == "COMPLETED":
                print("Job completed successfully!")
                output_data = status_result.get("output", "")
                
                # Different output formats depending on the RunPod setup
                print(f"Output data type: {type(output_data)}")
                print(f"Output data sample: {str(output_data)[:500]}")
                
                # Parse the output data into text
                output_text = self._parse_output_data(output_data)
                
                # Format in a way that matches the common response structure
                return self._create_response_object(job_id, output_text)
                
            elif status in ["FAILED", "CANCELLED"]:
                raise Exception(f"RunPod job {status}: {status_result}")
            
            # If still running or queued, increase wait time for next poll
            wait_time = min(wait_time * 2, 30)  # Exponential backoff, max 30 seconds
        
        # If we've exhausted all attempts
        raise Exception(f"RunPod job timed out after {max_attempts} polling attempts")
    
    async def _async_poll_for_results(self, job_id, headers):
        """Poll for results with exponential backoff (asynchronous)"""
        max_attempts = 10
        wait_time = 2  # Start with 2 seconds
        
        for attempt in range(max_attempts):
            print(f"Async polling attempt {attempt+1}/{max_attempts}. Waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            
            # Check job status
            status_url = f"{self.status_base}/{job_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    status_url,
                    headers=headers
                ) as status_response:
                    if status_response.status != 200:
                        text = await status_response.text()
                        print(f"Error checking job status: {status_response.status}, {text}")
                        wait_time = min(wait_time * 2, 30)  # Exponential backoff, max 30 seconds
                        continue
                    
                    status_result = await status_response.json()
                    status = status_result.get("status")
                    
                    print(f"Job status: {status}")
                    
                    if status == "COMPLETED":
                        print("Job completed successfully!")
                        output_data = status_result.get("output", "")
                        
                        # Different output formats depending on the RunPod setup
                        print(f"Output data type: {type(output_data)}")
                        print(f"Output data sample: {str(output_data)[:500]}")
                        
                        # Parse the output data into text
                        output_text = self._parse_output_data(output_data)
                        
                        # Format in a way that matches the common response structure
                        return self._create_response_object(job_id, output_text)
                        
                    elif status in ["FAILED", "CANCELLED"]:
                        raise Exception(f"RunPod job {status}: {status_result}")
            
            # If still running or queued, increase wait time for next poll
            wait_time = min(wait_time * 2, 30)  # Exponential backoff, max 30 seconds
        
        # If we've exhausted all attempts
        raise Exception(f"RunPod job timed out after {max_attempts} polling attempts")
    
    def _parse_output_data(self, output_data):
        """Parse the output data into text"""
        try:
            if isinstance(output_data, str):
                return output_data
            elif isinstance(output_data, dict):
                return output_data.get("text", str(output_data))
            elif isinstance(output_data, list) and len(output_data) > 0:
                # RunPod sometimes returns a list with a complex structure
                try:
                    # Try to extract from choices/tokens structure
                    if 'choices' in output_data[0] and isinstance(output_data[0]['choices'], list):
                        all_tokens = []
                        for choice in output_data[0]['choices']:
                            if 'tokens' in choice and isinstance(choice['tokens'], list):
                                all_tokens.extend(choice['tokens'])
                        return ''.join(all_tokens)
                    else:
                        # Fallback to string representation if we can't parse
                        return str(output_data)
                except (IndexError, KeyError, TypeError) as e:
                    print(f"Error parsing RunPod output: {e}")
                    return str(output_data)
            else:
                return str(output_data)
        except Exception as e:
            print(f"Error parsing output data: {e}")
            return str(output_data)
    
    def _create_response_object(self, job_id, output_text):
        """Create a standardized response object"""
        # Estimated token counts
        estimated_prompt_tokens = len(output_text.split()) * 1.3
        estimated_completion_tokens = len(output_text.split()) * 1.3
        
        # Create the response object
        response_data = {
            "id": f"runpod-{job_id}",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": "runpod/mistral",
            "usage": {
                "prompt_tokens": int(estimated_prompt_tokens),
                "completion_tokens": int(estimated_completion_tokens),
                "total_tokens": int(estimated_prompt_tokens + estimated_completion_tokens)
            }
        }
        
        # Return a ModelResponse object
        return ModelResponse(
            id=response_data["id"],
            choices=response_data["choices"],
            model=response_data["model"],
            usage=response_data["usage"]
        )
    
    def _generate_cache_key(self, prompt, temperature, max_tokens, kwargs):
        """Generate a cache key from the request parameters"""
        # Create a string representation of the kwargs for the cache key
        kwargs_str = ",".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        
        # Create a shortened prompt key using a hash
        prompt_hash = hash(prompt) % 10000000  # shortened hash
        
        return f"{prompt_hash}-{temperature}-{max_tokens}-{kwargs_str}"
    
    def _add_to_cache(self, key, value):
        """Add a response to the cache with LRU eviction"""
        # If cache is at max size, remove the oldest entry
        if len(self.response_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        # Add the new entry
        self.response_cache[key] = value
            
    def format_messages_to_prompt(self, messages):
        """Convert a list of messages to a single prompt string for Mistral 24B."""
        prompt = ""
        
        # Extract system message if present
        system_message = None
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
                break
        
        # Format conversation using Mistral's expected format
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            
            # Skip system messages as they'll be included with the first user message
            if role == "system":
                continue
                
            if role == "user":
                # If this is the first user message and we have a system message, include it
                if system_message and not any(m["role"] == "user" for m in messages[:i]):
                    prompt += f"<s>[INST] {system_message}\n\n{content} [/INST]</s>\n\n"
                else:
                    prompt += f"<s>[INST] {content} [/INST]</s>\n\n"
                    
            elif role == "assistant":
                prompt += f"{content}\n\n"
        
        # Add a final user message if the last message was from the assistant
        if messages[-1]["role"] == "assistant":
            prompt += "<s>[INST] Please continue. [/INST]</s>\n\n"
        
        return prompt
    
    async def async_streaming_completion(self, model, messages, temperature=0.7, max_tokens=2048, callback=None, **kwargs):
        """Stream tokens as they are generated (async).
        
        Note: RunPod API doesn't natively support streaming, so this is simulated by polling
        frequently and detecting new tokens.
        """
        # Implementation depends on how the RunPod API works
        # This is a simplified version that doesn't actually stream but returns the final result
        response = await self.async_completion(model, messages, temperature, max_tokens, **kwargs)
        
        if callback and response.choices:
            callback(response.choices[0].message.content)
            
        return response