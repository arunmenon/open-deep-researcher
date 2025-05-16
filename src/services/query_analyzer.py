from typing import List, Dict, Any
from ..models.strategy import ModelStrategy
from ..utils.response_parser import ResponseParser
from ..utils.response_models import BreadthDepthResponse, FollowUpQueriesResponse
from ..prompts import system_prompts, user_prompts

class QueryAnalyzer:
    """Service for analyzing queries and generating follow-up questions"""
    
    def __init__(self, model_strategy: ModelStrategy):
        """Initialize the query analyzer
        
        Args:
            model_strategy: The model strategy to use for analysis
        """
        self.model_strategy = model_strategy
    
    async def determine_research_parameters(self, query: str) -> BreadthDepthResponse:
        """Determine appropriate research breadth and depth based on query complexity.
        
        Args:
            query: The user's research query
            
        Returns:
            BreadthDepthResponse with breadth (1-10), depth (1-5), and explanation
        """
        # Get the prompt from the externalized prompts module
        user_prompt = user_prompts.research_parameters_prompt(query)

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": system_prompts.RESEARCH_PLANNER},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model strategy
            print("Determining research breadth and depth...")
            
            response = await self.model_strategy.execute_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=4096
            )
            
            output_text = response.choices[0].message.content
            
            # Parse the response using the ResponseParser
            try:
                return ResponseParser.parse_json_response(output_text, BreadthDepthResponse)
            except ValueError as e:
                print(f"Could not parse JSON from response: {e}")
                return BreadthDepthResponse(
                    breadth=4,
                    depth=2,
                    explanation=f"Default values used for research on '{query}'."
                )
            
        except Exception as e:
            print(f"Error determining research parameters: {e}")
            
            # Use default values
            print("Using default research parameters.")
            return BreadthDepthResponse(
                breadth=4,
                depth=2,
                explanation=f"Default values used for research on '{query}'."
            )
    
    async def generate_follow_up_questions(self, query: str, max_questions: int = 3) -> List[str]:
        """Generate follow-up questions to clarify research direction
        
        Args:
            query: The initial user query
            max_questions: Maximum number of questions to generate
            
        Returns:
            List of follow-up questions
        """
        # Get the prompt from the externalized prompts module
        user_prompt = user_prompts.follow_up_questions_prompt(query, max_questions)

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": system_prompts.FOLLOW_UP_GENERATOR},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model strategy
            print("Generating follow-up questions...")
            
            response = await self.model_strategy.execute_completion(
                messages=messages,
                temperature=1.0,
                max_tokens=4096
            )
            
            output_text = response.choices[0].message.content
            
            # Parse the response using the ResponseParser
            try:
                questions_response = ResponseParser.parse_json_response(output_text, FollowUpQueriesResponse)
                return questions_response.follow_up_queries
            except ValueError as e:
                print(f"Could not parse JSON for follow-up questions: {e}")
                # Generate some default questions if extraction fails
                return [
                    f"What specific aspects of {query} are you interested in?",
                    f"What is your goal for researching {query}?",
                    f"Any specific timeframe or context for {query}?"
                ][:max_questions]
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            
            # Generate some default questions
            print("Using default follow-up questions.")
            return [
                f"What specific aspects of {query} are you interested in?",
                f"What is your goal for researching {query}?", 
                f"Any specific timeframe or context for {query}?"
            ][:max_questions]