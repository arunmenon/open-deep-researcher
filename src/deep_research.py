from typing import Callable, List, TypeVar, Any, Dict, Optional, Union
import asyncio
import datetime
import json
import os
import uuid

import math
from pydantic import BaseModel

from dotenv import load_dotenv
import litellm
from litellm.utils import ModelResponse, Message
from google.genai import types

# Keep these imports for compatibility during transition
import google.generativeai as genai
from google import genai as genai_client
from google.ai.generativelanguage_v1beta.types import content


# Pydantic models for schema validation
class BreadthDepthResponse(BaseModel):
    breadth: int
    depth: int
    explanation: str

class FollowUpQueriesResponse(BaseModel):
    follow_up_queries: List[str]

class QueriesResponse(BaseModel):
    queries: List[str]

class QuerySimilarityResponse(BaseModel):
    are_similar: bool
    
class ProcessResultResponse(BaseModel):
    learnings: List[str]
    follow_up_questions: List[str]


class ResearchProgress:
    def __init__(self, depth: int, breadth: int):
        self.total_depth = depth
        self.total_breadth = breadth
        self.current_depth = depth
        self.current_breadth = 0
        self.queries_by_depth = {}
        self.query_order = []  # Track order of queries
        self.query_parents = {}  # Track parent-child relationships
        self.total_queries = 0  # Total number of queries including sub-queries
        self.completed_queries = 0
        self.query_ids = {}  # Store persistent IDs for queries
        self.root_query = None  # Store the root query

    def start_query(self, query: str, depth: int, parent_query: str = None):
        """Record the start of a new query"""
        if depth not in self.queries_by_depth:
            self.queries_by_depth[depth] = {}

        if query not in self.queries_by_depth[depth]:
            # Generate ID only once per query
            if query not in self.query_ids:
                self.query_ids[query] = str(uuid.uuid4())
                
            self.queries_by_depth[depth][query] = {
                "completed": False,
                "learnings": [],
                "id": self.query_ids[query]  # Use persistent ID
            }
            self.query_order.append(query)
            if parent_query:
                self.query_parents[query] = parent_query
            else:
                self.root_query = query  # Set as root if no parent
            self.total_queries += 1

        self.current_depth = depth
        self.current_breadth = len(self.queries_by_depth[depth])
        self._report_progress(f"Starting query: {query}")

    def add_learning(self, query: str, depth: int, learning: str):
        """Record a learning for a specific query"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if learning not in self.queries_by_depth[depth][query]["learnings"]:
                self.queries_by_depth[depth][query]["learnings"].append(learning)
                self._report_progress(f"Added learning for query: {query}")

    def complete_query(self, query: str, depth: int):
        """Mark a query as completed"""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if not self.queries_by_depth[depth][query]["completed"]:
                self.queries_by_depth[depth][query]["completed"] = True
                self.completed_queries += 1
                self._report_progress(f"Completed query: {query}")

                # Check if parent query exists and update its status if all children are complete
                parent_query = self.query_parents.get(query)
                if parent_query:
                    self._update_parent_status(parent_query)

    def _update_parent_status(self, parent_query: str):
        """Update parent query status based on children completion"""
        # Find all children of this parent
        children = [q for q, p in self.query_parents.items() if p == parent_query]
        
        # Check if all children are complete
        parent_depth = next((d for d, queries in self.queries_by_depth.items() 
                           if parent_query in queries), None)
        
        if parent_depth is not None:
            all_children_complete = all(
                self.queries_by_depth[d][q]["completed"]
                for q in children
                for d in self.queries_by_depth
                if q in self.queries_by_depth[d]
            )
            
            if all_children_complete:
                # Complete the parent query
                self.complete_query(parent_query, parent_depth)

    def _report_progress(self, action: str):
        """Report current progress"""
        print(f"\nResearch Progress Update:")
        print(f"Action: {action}")

        # Build and print the tree starting from the root query
        if self.root_query:
            tree = self._build_research_tree()
            print("\nQuery Tree Structure:")
            print(json.dumps(tree, indent=2))

        print(f"\nOverall Progress: {self.completed_queries}/{self.total_queries} queries completed")
        print("")

    def _build_research_tree(self):
        """Build the full research tree structure"""
        def build_node(query):
            """Recursively build the tree node"""
            # Find the depth for this query
            depth = next((d for d, queries in self.queries_by_depth.items() 
                         if query in queries), 0)
            
            data = self.queries_by_depth[depth][query]
            
            # Find all children of this query
            children = [q for q, p in self.query_parents.items() if p == query]
            
            return {
                "query": query,
                "id": self.query_ids[query],
                "status": "completed" if data["completed"] else "in_progress",
                "depth": depth,
                "learnings": data["learnings"],
                "sub_queries": [build_node(child) for child in children],
                "parent_query": self.query_parents.get(query)
            }

        # Start building from the root query
        if self.root_query:
            return build_node(self.root_query)
        return {}

    def get_learnings_by_query(self):
        """Get all learnings organized by query"""
        learnings = {}
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                if data["learnings"]:
                    learnings[query] = data["learnings"]
        return learnings


load_dotenv()

class RunpodProvider:
    """Custom provider for RunPod API."""
    
    def __init__(self, api_key, endpoint_id):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.api_base = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.status_base = f"https://api.runpod.ai/v2/{endpoint_id}/status"
        
    def completion(self, model, messages, temperature=0.7, max_tokens=2048, **kwargs):
        """Make a completion request to RunPod API."""
        import requests
        import json
        import uuid
        import time
        
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
        
        # Format in a way that matches the litellm response structure
        litellm_response = {
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
        
        # Create a ModelResponse object using the litellm converter
        try:
            from litellm.utils import convert_to_model_response_object
            return convert_to_model_response_object(litellm_response)
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
                    
            return SimpleModelResponse(litellm_response)
            
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


class DeepSearch:
    def __init__(self, api_key: str = None, mode: str = "balanced", use_mistral: bool = True, runpod_api_key: str = None):
        """
        Initialize DeepSearch with a mode parameter:
        - "fast": Prioritizes speed (reduced breadth/depth, highest concurrency)
        - "balanced": Default balance of speed and comprehensiveness
        - "comprehensive": Maximum detail and coverage

        In this version, we use exclusively Mistral on RunPod.
        """
        self.runpod_api_key = runpod_api_key or os.getenv("RUNPOD_API_KEY")
        if not self.runpod_api_key:
            raise ValueError("RunPod API key is required")
            
        self.query_history = set()
        self.mode = mode
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "w0oa6hyd2q40jw")  # Get from env or use default
        
        # Create a custom RunPod provider for direct API access
        self.runpod_provider = RunpodProvider(self.runpod_api_key, self.endpoint_id)
        # For compatibility with existing code that might reference this attribute
        self.runpod_adapter = self.runpod_provider
        
        # Configure the model name
        self.litellm_model_name = "runpod/mistral-24b-instruct" 
        
        # Register our custom completion function with LiteLLM
        def custom_completion(model, messages, **kwargs):
            return self.runpod_provider.completion(model, messages, **kwargs)
        
        # Add our custom provider to litellm
        try:
            # Check if register_model method exists in current litellm version
            if hasattr(litellm, 'register_model'):
                litellm.register_model(self.litellm_model_name, custom_completion)
            # Check if register_completion_function method exists
            elif hasattr(litellm, 'register_completion_function'):
                litellm.register_completion_function(
                    model_name=self.litellm_model_name,
                    completion_function=custom_completion
                )
            else:
                print("Warning: Could not register custom model with litellm. Using a direct approach.")
        except Exception as e:
            print(f"Warning: Could not register custom model with litellm ({str(e)}). Using a direct approach.")
                
        print("Configured LiteLLM to use Mistral on RunPod")

    def determine_research_breadth_and_depth(self, query: str) -> BreadthDepthResponse:
        user_prompt = f"""
			You are a research planning assistant. Your task is to determine the appropriate breadth and depth for researching a topic defined by a user's query. Evaluate the query's complexity and scope, then recommend values on the following scales:

			Breadth: Scale of 1 (very narrow) to 10 (extensive, multidisciplinary).
			Depth: Scale of 1 (basic overview) to 5 (highly detailed, in-depth analysis).
			Defaults:

			Breadth: 4
			Depth: 2
			Note: More complex or "harder" questions should prompt higher ratings on one or both scales, reflecting the need for more extensive research and deeper analysis.

			Response Format:
			Output your recommendation in JSON format, including an explanation. For example:
			```json
			{{
				"breadth": 4,
				"depth": 2,
				"explanation": "The topic is moderately complex; a broad review is needed (breadth 4) with a basic depth analysis (depth 2)."
			}}
			```

			Here is the user's query:
			<query>{query}</query>
			"""

        # Define messages for LiteLLM
        messages = [
            {"role": "system", "content": "You are a research planning assistant that determines appropriate research parameters."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make a direct API call to RunPod instead of using LiteLLM
            print("Making direct API call to RunPod...")
            
            prompt = self.runpod_adapter.format_messages_to_prompt(messages)
            
            import requests
            import json
            
            payload = {
                "input": {
                    "prompt": prompt,
                    "max_tokens": 8192,
                    "temperature": 1.0,
                    "top_p": 0.95
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.runpod_api_key}"
            }
            
            response = requests.post(
                f"https://api.runpod.ai/v2/{self.endpoint_id}/run",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"RunPod API error: {response.status_code}, {response.text}")
            
            result = response.json()
            output_text = result.get("output", "")
            
            # Try to extract JSON from the response text
            try:
                # Look for JSON pattern in the text
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', output_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no JSON block found, try to parse the whole text
                    json_str = output_text
                
                # Parse JSON
                json_response = json.loads(json_str)
                
                # Make sure required fields are present
                if not all(k in json_response for k in ["breadth", "depth", "explanation"]):
                    # If not, set default values
                    json_response = {
                        "breadth": 4,
                        "depth": 2,
                        "explanation": f"Default values used for research on '{query}'."
                    }
                
                return BreadthDepthResponse(**json_response)
                
            except json.JSONDecodeError:
                # If we can't parse JSON, use default values
                print(f"Could not parse JSON from response. Using default values.")
                return BreadthDepthResponse(
                    breadth=4,
                    depth=2,
                    explanation=f"Default values used for research on '{query}'."
                )
            
        except Exception as e:
            print(f"Error calling RunPod API: {e}")
            
            # Use default values instead of trying Gemini
            print("Using default research parameters.")
            return BreadthDepthResponse(
                breadth=4,
                depth=2,
                explanation=f"Default values used for research on '{query}'."
            )

    def generate_follow_up_questions(
        self,  # Changed to instance method
        query: str,
        max_questions: int = 3,
    ) -> List[str]:
        import re
        import json
        
        user_prompt = """
			Given the following query from the user, ask some follow up questions to clarify the research direction.

			Return a maximum of {} questions, but feel free to return less if the original query is clear: <query>{}</query>
        
        Your response should be in JSON format as follows:
        {{
          "follow_up_queries": [
            "Question 1?",
            "Question 2?",
            "Question 3?"
          ]
        }}
			""".format(max_questions, query)

        # Define messages for direct API call
        messages = [
            {"role": "system", "content": "You are a research assistant that generates follow-up questions to clarify research direction."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Use direct API call to RunPod instead of LiteLLM
            print("Making direct API call to RunPod for follow-up questions...")
            
            # Use the RunPod provider directly
            response = self.runpod_provider.completion(
                model="mistral-24b-instruct",  # Model name doesn't matter for direct provider
                messages=messages,
                temperature=1.0,
                top_p=0.95,
                max_tokens=4096
            )
            
            output_text = response.choices[0].message.content
            
            # Try to extract JSON from the response text
            try:
                # Look for JSON pattern in the text
                json_match = re.search(r'```json\s*(.*?)\s*```', output_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no JSON block found, try to parse the whole text
                    json_str = output_text
                
                json_response = json.loads(json_str)
                
                # Make sure follow_up_queries field is present
                if "follow_up_queries" not in json_response:
                    # Try to extract questions from the text if no JSON
                    questions = re.findall(r'\d+\.\s+(.*?\?)', output_text)
                    if questions:
                        return questions[:max_questions]
                    else:
                        # Return default questions
                        return [
                            f"What specific aspects of {query} are you interested in?",
                            f"What is your goal for researching {query}?",
                            f"Any specific timeframe or context for {query}?"
                        ][:max_questions]
                
                return json_response["follow_up_queries"]
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Could not parse JSON for follow-up questions: {e}")
                # Try to extract questions directly from text
                questions = re.findall(r'\d+\.\s+(.*?\?)', output_text)
                if questions:
                    return questions[:max_questions]
                else:
                    # Generate some default questions if extraction fails
                    return [
                        f"What specific aspects of {query} are you interested in?",
                        f"What is your goal for researching {query}?",
                        f"Any specific timeframe or context for {query}?"
                    ][:max_questions]
            
        except Exception as e:
            print(f"Error calling RunPod API for follow-up questions: {e}")
            
            # Generate some default questions
            print("Using default follow-up questions.")
            return [
                f"What specific aspects of {query} are you interested in?",
                f"What is your goal for researching {query}?", 
                f"Any specific timeframe or context for {query}?"
            ][:max_questions]

    def generate_queries(
            self,
            query: str,
            num_queries: int = 3,
            learnings: list[str] = [],
            previous_queries: set[str] = None  # Add previous_queries parameter
    ) -> List[str]:
        import re
        import json
        
        now = datetime.datetime.now().strftime("%Y-%m-%d")

        # Format previous queries for the prompt
        previous_queries_text = ""
        if previous_queries:
            previous_queries_text = "\n\nPreviously asked queries (avoid generating similar ones):\n" + \
                "\n".join([f"- {q}" for q in previous_queries])

        user_prompt = f"""
        Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a maximum
        of {num_queries} queries, but feel free to return less if the original prompt is clear.

        IMPORTANT: Each query must be unique and significantly different from both each other AND the previously asked queries.
        Avoid semantic duplicates or queries that would likely return similar information.

        Original prompt: <prompt>${query}</prompt>
        {previous_queries_text}
        
        Your response should be in JSON format as follows:
        {{
          "queries": [
            "Query 1",
            "Query 2",
            "Query 3"
          ]
        }}
        """

        learnings_prompt = "" if not learnings else "Here are some learnings from previous research, use them to generate more specific queries: " + \
            "\n".join(learnings)
            
        full_prompt = user_prompt + learnings_prompt

        # Define messages for direct API call
        messages = [
            {"role": "system", "content": "You are a research assistant that generates search queries for research topics."},
            {"role": "user", "content": full_prompt}
        ]

        try:
            # Use direct RunPod API 
            print("Making direct API call to RunPod for generating queries...")
            
            response = self.runpod_provider.completion(
                model="mistral-24b-instruct",  # Model name doesn't matter for direct provider
                messages=messages,
                temperature=1.0,
                top_p=0.95,
                max_tokens=4096
            )
            
            output_text = response.choices[0].message.content
            
            # Try to extract JSON from the response text
            try:
                # Look for JSON pattern in the text
                json_match = re.search(r'```json\s*(.*?)\s*```', output_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no JSON block found, try to parse the whole text
                    json_str = output_text
                
                json_response = json.loads(json_str)
                
                # Make sure queries field is present
                if "queries" not in json_response:
                    # Try to extract queries from the text if no JSON
                    lines = re.findall(r'\d+\.\s+(.*)', output_text)
                    if lines:
                        return lines[:num_queries]
                    else:
                        # Generate default queries
                        return [
                            f"{query} research papers",
                            f"{query} latest developments",
                            f"{query} technical details"
                        ][:num_queries]
                
                return json_response["queries"]
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Could not parse JSON for query generation: {e}")
                # Try to extract queries directly from text
                lines = re.findall(r'\d+\.\s+(.*)', output_text)
                if lines:
                    return lines[:num_queries]
                else:
                    # Generate some default queries if extraction fails
                    return [
                        f"{query} research papers",
                        f"{query} latest developments",
                        f"{query} technical details"
                    ][:num_queries]
            
        except Exception as e:
            print(f"Error calling LiteLLM for query generation: {e}")
            
            # Generate some default queries
            print("Using default queries.")
            return [
                f"{query} research papers",
                f"{query} latest developments",
                f"{query} technical details"
            ][:num_queries]

    def format_text_with_sources(self, response_dict: dict, answer: str):
        """
        Format text with sources from Gemini response, adding citations at specified positions.
        Returns tuple of (formatted_text, sources_dict).
        """
        if not response_dict or not response_dict.get('candidates'):
            return answer, {}

        # Get grounding metadata from the response
        grounding_metadata = response_dict['candidates'][0].get(
            'grounding_metadata')
        if not grounding_metadata:
            return answer, {}

        # Get grounding chunks and supports
        grounding_chunks = grounding_metadata.get('grounding_chunks', [])
        grounding_supports = grounding_metadata.get('grounding_supports', [])

        if not grounding_chunks or not grounding_supports:
            return answer, {}

        try:
            # Create mapping of URLs
            sources = {
                i: {
                    'link': chunk.get('web', {}).get('uri', ''),
                    'title': chunk.get('web', {}).get('title', '')
                }
                for i, chunk in enumerate(grounding_chunks)
                if chunk.get('web')
            }

            # Create a list of (position, citation) tuples
            citations = []
            for support in grounding_supports:
                segment = support.get('segment', {})
                indices = support.get('grounding_chunk_indices', [])

                if indices and segment and segment.get('end_index') is not None:
                    end_index = segment['end_index']
                    source_idx = indices[0]
                    if source_idx in sources:
                        citation = f"[[{source_idx + 1}]]({sources[source_idx]['link']})"
                        citations.append((end_index, citation))

            # Sort citations by position (end_index)
            citations.sort(key=lambda x: x[0])

            # Insert citations into the text
            result = ""
            last_pos = 0
            for pos, citation in citations:
                result += answer[last_pos:pos]
                result += citation
                last_pos = pos

            # Add any remaining text
            result += answer[last_pos:]

            return result, sources

        except Exception as e:
            print(f"Error processing grounding metadata: {e}")
            return answer, {}

    def search(self, query: str):
        """
        Perform a search using Mistral on RunPod.
        This simpler implementation doesn't use search tools but asks the model to generate information.
        """
        try:
            # Create system and user messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides comprehensive research information on topics. Please provide detailed and accurate information about the following query, including relevant facts, figures, dates, and analysis."},
                {"role": "user", "content": f"Please research and provide detailed information about: {query}"}
            ]
            
            # Use direct RunPod API
            print(f"Making direct API call to RunPod for search query: {query}")
            
            # Call RunPod provider directly
            response = self.runpod_provider.completion(
                model="mistral-24b-instruct",  # Model name doesn't matter for direct provider
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Create empty sources since we're not using a search tool
            sources = {}
            
            return response_text, sources
            
        except Exception as e:
            print(f"Error performing search with RunPod: {e}")
            # Return empty results in case of failure
            return f"Error performing search: {e}", {}

    async def process_result(
        self,
        query: str,
        result: str,
        num_learnings: int = 3,
        num_follow_up_questions: int = 3,
    ) -> Dict[str, List[str]]:
        import re
        import json
        
        print(f"Processing result for query: {query}")

        user_prompt = f"""
			Given the following result from a SERP search for the query <query>{query}</query>, generate a list of learnings from the result. Return a maximum of {num_learnings} learnings, but feel free to return less if the result is clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.
        
        Here is the result:
        {result}
        
        Your response should be in JSON format as follows:
        {{
          "learnings": [
            "Learning 1",
            "Learning 2",
            "Learning 3"
          ],
          "follow_up_questions": [
            "Question 1?",
            "Question 2?",
            "Question 3?"
          ]
        }}
			"""

        # Define messages for direct API call
        messages = [
            {"role": "system", "content": "You extract key learnings and generate follow-up questions from search results."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Use direct RunPod API
            print("Making direct API call to RunPod for processing results...")
            
            response = self.runpod_provider.completion(
                model="mistral-24b-instruct",  # Model name doesn't matter for direct provider
                messages=messages,
                temperature=1.0,
                top_p=0.95,
                max_tokens=4096
            )
            
            output_text = response.choices[0].message.content
            
            # Try to extract JSON from the response text
            try:
                # Look for JSON pattern in the text
                json_match = re.search(r'```json\s*(.*?)\s*```', output_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no JSON block found, try to parse the whole text
                    json_str = output_text
                
                json_response = json.loads(json_str)
                
                # Make sure required fields are present
                if not all(k in json_response for k in ["learnings", "follow_up_questions"]):
                    # If missing fields, try to extract directly from text
                    learnings = []
                    questions = []
                    
                    # Try to extract learnings
                    learnings_section = re.search(r'Learnings:(.*?)(?:Follow-up Questions:|$)', output_text, re.DOTALL | re.IGNORECASE)
                    if learnings_section:
                        learnings = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|\n\n|$)', learnings_section.group(1), re.DOTALL)
                        learnings = [l.strip() for l in learnings if l.strip()]
                    
                    # Try to extract questions
                    questions_section = re.search(r'Follow-up Questions:(.*?)(?:\n\n|$)', output_text, re.DOTALL | re.IGNORECASE)
                    if questions_section:
                        questions = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|\n\n|$)', questions_section.group(1), re.DOTALL)
                        questions = [q.strip() for q in questions if q.strip()]
                    
                    # Use defaults if extraction failed
                    if not learnings:
                        learnings = [f"Key information about {query}"]
                    if not questions:
                        questions = [f"What are the most important aspects of {query}?"]
                    
                    return {
                        "learnings": learnings[:num_learnings],
                        "follow_up_questions": questions[:num_follow_up_questions]
                    }
                
                # Process results
                learnings = json_response["learnings"][:num_learnings]
                follow_up_questions = json_response["follow_up_questions"][:num_follow_up_questions]
                
                print(f"Results from {query}:")
                print(f"Learnings: {learnings}\n")
                print(f"Follow up questions: {follow_up_questions}\n")
                
                return {
                    "learnings": learnings,
                    "follow_up_questions": follow_up_questions
                }
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Could not parse JSON for result processing: {e}")
                # Try to extract directly from text
                learnings = []
                questions = []
                
                # Try to extract learnings
                learnings_section = re.search(r'Learnings:(.*?)(?:Follow-up Questions:|$)', output_text, re.DOTALL | re.IGNORECASE)
                if learnings_section:
                    learnings = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|\n\n|$)', learnings_section.group(1), re.DOTALL)
                    learnings = [l.strip() for l in learnings if l.strip()]
                
                # Try to extract questions
                questions_section = re.search(r'Follow-up Questions:(.*?)(?:\n\n|$)', output_text, re.DOTALL | re.IGNORECASE)
                if questions_section:
                    questions = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|\n\n|$)', questions_section.group(1), re.DOTALL)
                    questions = [q.strip() for q in questions if q.strip()]
                
                # Use defaults if extraction failed
                if not learnings:
                    learnings = [f"Key information about {query}"]
                if not questions:
                    questions = [f"What are the most important aspects of {query}?"]
                
                print(f"Results from {query}:")
                print(f"Learnings: {learnings}\n")
                print(f"Follow up questions: {questions}\n")
                
                return {
                    "learnings": learnings[:num_learnings],
                    "follow_up_questions": questions[:num_follow_up_questions]
                }
            
        except Exception as e:
            print(f"Error calling LiteLLM for result processing: {e}")
            
            # Use default values
            default_learnings = [f"Key information about {query}"]
            default_questions = [f"What are the most important aspects of {query}?"]
            
            print(f"Using default learnings and questions for {query}")
            
            return {
                "learnings": default_learnings,
                "follow_up_questions": default_questions
            }

    def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """Helper method to check if two queries are semantically similar using direct RunPod API"""
        user_prompt = f"""
        Compare these two search queries and determine if they are semantically similar 
        (i.e., would likely return similar search results or are asking about the same topic):

        Query 1: {query1}
        Query 2: {query2}

        Consider:
        1. Key concepts and entities
        2. Intent of the queries
        3. Scope and specificity
        4. Core topic overlap

        Only respond with true if the queries are notably similar, false otherwise.
        
        Your response should be in JSON format as follows:
        {{
          "are_similar": true or false
        }}
        """

        # Define messages for direct API call
        messages = [
            {"role": "system", "content": "You determine whether search queries are semantically similar."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Call RunPod provider directly
            response = self.runpod_provider.completion(
                model="mistral-24b-instruct",  # Model name doesn't matter for direct provider
                messages=messages,
                temperature=0.1,  # Low temperature for more consistent results
                top_p=0.95,
                max_tokens=4096
            )

            # Extract and parse the response
            json_response = json.loads(response.choices[0].message.content)
            return QuerySimilarityResponse(**json_response).are_similar
            
        except Exception as e:
            print(f"Error calling LiteLLM for query similarity: {e}")
            
            # Fallback to legacy approach if needed
            generation_config = {
                "temperature": 0.1,  # Low temperature for more consistent results
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
                "response_schema": content.Schema(
                    type=content.Type.OBJECT,
                    required=["are_similar"],
                    properties={
                        "are_similar": content.Schema(
                            type=content.Type.BOOLEAN,
                            description="True if queries are semantically similar, false otherwise"
                        )
                    }
                )
            }

            try:
                model = genai.GenerativeModel(
                    "gemini-2.0-flash",
                    generation_config=generation_config,
                )

                response = model.generate_content(user_prompt)
                answer = json.loads(response.text)
                return answer["are_similar"]
            except Exception as e:
                print(f"Error comparing queries: {str(e)}")
                # In case of error, assume queries are different to avoid missing potentially unique results
                return False

    async def deep_research(self, query: str, breadth: int, depth: int, learnings: list[str] = [], visited_urls: dict[int, dict] = {}, parent_query: str = None):
        progress = ResearchProgress(depth, breadth)
        
        # Start the root query
        progress.start_query(query, depth, parent_query)

        # Adjust number of queries based on mode
        max_queries = {
            "fast": 3,
            "balanced": 7,
            "comprehensive": 5 # kept lower than balanced due to recursive multiplication
        }[self.mode]

        queries = self.generate_queries(
            query,
            min(breadth, max_queries),
            learnings,
            previous_queries=self.query_history
        )

        self.query_history.update(queries)
        unique_queries = list(queries)[:breadth]

        async def process_query(query_str: str, current_depth: int, parent: str = None):
            try:
                # Start this query as a sub-query of the parent
                progress.start_query(query_str, current_depth, parent)

                result = self.search(query_str)
                processed_result = await self.process_result(
                    query=query_str,
                    result=result[0],
                    num_learnings=min(3, math.ceil(breadth / 2)),
                    num_follow_up_questions=min(2, math.ceil(breadth / 2))
                )

                # Record learnings
                for learning in processed_result["learnings"]:
                    progress.add_learning(query_str, current_depth, learning)

                new_urls = result[1]
                max_idx = max(visited_urls.keys()) if visited_urls else -1
                all_urls = {
                    **visited_urls,
                    **{(i + max_idx + 1): url_data for i, url_data in new_urls.items()}
                }

                # Only go deeper if in comprehensive mode and depth > 1
                if self.mode == "comprehensive" and current_depth > 1:
                    # Reduced breadth for deeper levels
                    new_breadth = min(2, math.ceil(breadth / 2))
                    new_depth = current_depth - 1

                    # Select most important follow-up question instead of using all
                    if processed_result['follow_up_questions']:
                        # Take only the most relevant question
                        next_query = processed_result['follow_up_questions'][0]
                        
                        # Process the sub-query
                        sub_results = await process_query(
                            next_query,
                            new_depth,
                            query_str  # Pass current query as parent
                        )

                progress.complete_query(query_str, current_depth)
                return {
                    "learnings": processed_result["learnings"],
                    "visited_urls": all_urls
                }

            except Exception as e:
                print(f"Error processing query {query_str}: {str(e)}")
                progress.complete_query(query_str, current_depth)
                return {
                    "learnings": [],
                    "visited_urls": {}
                }

        # Process queries concurrently
        tasks = [process_query(q, depth, query) for q in unique_queries]
        results = await asyncio.gather(*tasks)

        # Combine results
        all_learnings = list(set(
            learning
            for result in results
            for learning in result["learnings"]
        ))

        all_urls = {}
        current_idx = 0
        seen_urls = set()
        for result in results:
            for url_data in result["visited_urls"].values():
                if url_data['link'] not in seen_urls:
                    all_urls[current_idx] = url_data
                    seen_urls.add(url_data['link'])
                    current_idx += 1

        # Complete the root query after all sub-queries are done
        progress.complete_query(query, depth)

        # save the tree structure to a json file
        with open("research_tree.json", "w") as f:
            json.dump(progress._build_research_tree(), f)

        return {
            "learnings": all_learnings,
            "visited_urls": all_urls
        }

    def generate_final_report(self, query: str, learnings: list[str], visited_urls: dict[int, dict]) -> str:
        # Format sources and learnings for the prompt
        sources_text = "\n".join([
            f"- {data['title']}: {data['link']}"
            for data in visited_urls.values()
        ])
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])

        user_prompt = f"""
        You are a creative research analyst tasked with synthesizing findings into an engaging and informative report.
        Create a comprehensive research report (minimum 3000 words) based on the following query and findings.
        
        Original Query: {query}
        
        Key Findings:
        {learnings_text}
        
        Sources Used:
        {sources_text}
        
        Guidelines:
        1. Design a creative and engaging report structure that best fits the content and topic
        2. Feel free to use any combination of:
           - Storytelling elements
           - Case studies
           - Scenarios
           - Visual descriptions
           - Analogies and metaphors
           - Creative section headings
           - Thought experiments
           - Future projections
           - Historical parallels
        3. Make the report engaging while maintaining professionalism
        4. Include all relevant data points but present them in an interesting way
        5. Structure the information in whatever way makes the most logical sense for this specific topic
        6. Feel free to break conventional report formats if a different approach would be more effective
        7. Consider using creative elements like:
           - "What if" scenarios
           - Day-in-the-life examples
           - Before/After comparisons
           - Expert perspectives
           - Trend timelines
           - Problem-solution frameworks
           - Impact matrices
        
        Requirements:
        - Minimum 3000 words
        - Must include all key findings and data points
        - Must maintain factual accuracy
        - Must be well-organized and easy to follow
        - Must include clear conclusions and insights
        - Must cite sources appropriately
        
        Be bold and creative in your approach while ensuring the report effectively communicates all the important information!
        """

        # Define messages for direct API call
        messages = [
            {"role": "system", "content": "You are a creative research analyst that synthesizes findings into engaging reports."},
            {"role": "user", "content": user_prompt}
        ]

        print("Generating final report...\n")

        try:
            # Using direct RunPod API
            print("Making direct API call to RunPod for final report generation...")
            
            response = self.runpod_provider.completion(
                model="mistral-24b-instruct",  # Model name doesn't matter for direct provider
                messages=messages,
                temperature=0.9,  # Increased for more creativity
                top_p=0.95,
                max_tokens=8192
            )
            
            # Extract the response text from the actual RunPod response format
            try:
                # Check if the response has proper structure
                if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                    formatted_text = response.choices[0].message.content
                else:
                    # For our custom objects or direct access to the RunPod output
                    if isinstance(response, str):
                        formatted_text = response
                    elif isinstance(response, list) and len(response) > 0:
                        # Extract from RunPod token format
                        if 'choices' in response[0] and isinstance(response[0]['choices'], list):
                            all_tokens = []
                            for choice in response[0]['choices']:
                                if 'tokens' in choice and isinstance(choice['tokens'], list):
                                    all_tokens.extend(choice['tokens'])
                            formatted_text = ''.join(all_tokens)
                        else:
                            formatted_text = str(response)
                    else:
                        # Some other object form
                        formatted_text = str(response)
                        
                # If no response, use a default message
                if not formatted_text:
                    formatted_text = f"Error: No content generated for report on {query}."
                    
                print(f"Final report content (first 500 chars):\n{formatted_text[:500]}...")
                
            except Exception as e:
                print(f"Error extracting content from response: {e}")
                formatted_text = f"Error processing report response for query on {query}: {str(e)}"
            
        except Exception as e:
            print(f"Error calling LiteLLM for final report: {e}")
            
            # Use basic placeholder text
            formatted_text = f"""
# Report on {query}

## Summary
This is a placeholder report. The report generation experienced technical difficulties.

## Key Findings
{learnings_text}

## Conclusion
Please try regenerating this report or consider refining your query.
"""

        # Add sources section
        sources_section = "\n# Sources\n" + "\n".join([
            f"- [{data['title']}]({data['link']})"
            for data in visited_urls.values()
        ])

        return formatted_text + sources_section

