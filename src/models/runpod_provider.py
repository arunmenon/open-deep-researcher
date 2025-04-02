import datetime
import requests
import json
import time
from typing import Dict, List, Any

from .provider import ModelProvider

class RunpodProvider(ModelProvider):
    """Provider for RunPod API."""
    
    def __init__(self, api_key, endpoint_id):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.api_base = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.status_base = f"https://api.runpod.ai/v2/{endpoint_id}/status"
        
    def completion(self, model, messages, temperature=0.7, max_tokens=2048, **kwargs):
        """Make a completion request to RunPod API."""
        # Format the messages for Mistral
        formatted_prompt = self.format_messages_to_prompt(messages)
        
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
                
                if isinstance(output_data, str):
                    output_text = output_data
                elif isinstance(output_data, dict):
                    output_text = output_data.get("text", str(output_data))
                elif isinstance(output_data, list) and len(output_data) > 0:
                    # RunPod sometimes returns a list with a complex structure
                    try:
                        # Try to extract from choices/tokens structure
                        if 'choices' in output_data[0] and isinstance(output_data[0]['choices'], list):
                            all_tokens = []
                            for choice in output_data[0]['choices']:
                                if 'tokens' in choice and isinstance(choice['tokens'], list):
                                    all_tokens.extend(choice['tokens'])
                            output_text = ''.join(all_tokens)
                        else:
                            # Fallback to string representation if we can't parse
                            output_text = str(output_data)
                    except (IndexError, KeyError, TypeError) as e:
                        print(f"Error parsing RunPod output: {e}")
                        output_text = str(output_data)
                else:
                    output_text = str(output_data)
                    
                break
            elif status in ["FAILED", "CANCELLED"]:
                raise Exception(f"RunPod job {status}: {status_result}")
            
            # If still running or queued, increase wait time for next poll
            wait_time = min(wait_time * 2, 30)  # Exponential backoff, max 30 seconds
        else:
            # If we've exhausted all attempts
            raise Exception(f"RunPod job timed out after {max_attempts} polling attempts")
        
        print(f"Output text received (first 100 chars): {output_text[:100]}...")
        
        # Format in a way that matches the common response structure
        response_data = {
            "id": f"runpod-{job_id}",
            "object": "chat.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": model,
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
            "usage": {
                "prompt_tokens": len(formatted_prompt) // 4,  # Rough estimate
                "completion_tokens": len(output_text) // 4,  # Rough estimate
                "total_tokens": (len(formatted_prompt) + len(output_text)) // 4  # Rough estimate
            }
        }
        
        # Try to convert to a common response object format
        try:
            from litellm.utils import convert_to_model_response_object
            return convert_to_model_response_object(response_data)
        except Exception as e:
            print(f"Warning: Could not convert to ModelResponse object: {e}")
            
            # Create a simple compatible object if conversion fails
            class SimpleModelResponse:
                def __init__(self, response_dict):
                    self.id = response_dict["id"]
                    self.choices = [SimpleChoice(choice) for choice in response_dict["choices"]]
                    
            class SimpleChoice:
                def __init__(self, choice_dict):
                    self.index = choice_dict["index"]
                    self.message = SimpleMessage(choice_dict["message"])
                    self.finish_reason = choice_dict["finish_reason"]
                    
            class SimpleMessage:
                def __init__(self, message_dict):
                    self.role = message_dict["role"]
                    self.content = message_dict["content"]
                    
            return SimpleModelResponse(response_data)
            
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