from typing import Dict, List, Any
import re

from ..models.strategy import ModelStrategy
from ..utils.response_parser import ResponseParser
from ..utils.response_models import ProcessResultResponse
from ..prompts import system_prompts, user_prompts

class ResultProcessor:
    """Service for processing search results"""
    
    def __init__(self, model_strategy: ModelStrategy):
        """Initialize the result processor
        
        Args:
            model_strategy: The model strategy to use
        """
        self.model_strategy = model_strategy
    
    async def process_result(self, query: str, result: str, num_learnings: int = 3, num_follow_up_questions: int = 3) -> Dict[str, List[str]]:
        """Process search results to extract key learnings and follow-up questions
        
        Args:
            query: The search query
            result: The search result text
            num_learnings: Maximum number of learnings to extract
            num_follow_up_questions: Maximum number of follow-up questions to generate
            
        Returns:
            Dictionary with 'learnings' and 'follow_up_questions' lists
        """
        print(f"Processing result for query: {query}")

        # Get the prompt from the externalized prompts module
        user_prompt = user_prompts.process_result_prompt(
            query, result, num_learnings, num_follow_up_questions)

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": system_prompts.RESULT_PROCESSOR},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model strategy
            print("Processing search results...")
            
            response = await self.model_strategy.execute_completion(
                messages=messages,
                temperature=1.0,
                max_tokens=4096
            )
            
            output_text = response.choices[0].message.content
            
            # Use ResponseParser to extract the structured data
            try:
                processed_result = ResponseParser.parse_json_response(output_text, ProcessResultResponse)
                
                # Limit to the requested number of items
                learnings = processed_result.learnings[:num_learnings]
                follow_up_questions = processed_result.follow_up_questions[:num_follow_up_questions]
                
                print(f"Results from {query}:")
                print(f"Learnings: {learnings}\n")
                print(f"Follow up questions: {follow_up_questions}\n")
                
                return {
                    "learnings": learnings,
                    "follow_up_questions": follow_up_questions
                }
                
            except ValueError as e:
                print(f"Could not parse JSON for result processing: {e}")
                # Use default values
                default_learnings = [f"Key information about {query}"]
                default_questions = [f"What are the most important aspects of {query}?"]
                
                print(f"Using default learnings and questions for {query}")
                
                return {
                    "learnings": default_learnings,
                    "follow_up_questions": default_questions
                }
            
        except Exception as e:
            print(f"Error processing search results: {e}")
            
            # Use default values
            default_learnings = [f"Key information about {query}"]
            default_questions = [f"What are the most important aspects of {query}?"]
            
            print(f"Using default learnings and questions for {query}")
            
            return {
                "learnings": default_learnings,
                "follow_up_questions": default_questions
            }