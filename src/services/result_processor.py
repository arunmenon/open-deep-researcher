from typing import Dict, List, Any
import re

from ..models.provider import ModelProvider
from ..utils.response_parser import ResponseParser
from ..utils.response_models import ProcessResultResponse

class ResultProcessor:
    """Service for processing search results"""
    
    def __init__(self, model_provider: ModelProvider, model_name: str):
        """Initialize the result processor
        
        Args:
            model_provider: The model provider to use
            model_name: The name of the model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
    
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

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": "You extract key learnings and generate follow-up questions from search results."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model provider
            print("Processing search results...")
            
            response = await self.model_provider.async_completion(
                model=self.model_name,
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