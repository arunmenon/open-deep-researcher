from typing import List, Dict, Any
from ..models.provider import ModelProvider
from ..utils.response_parser import ResponseParser
from ..utils.response_models import BreadthDepthResponse, FollowUpQueriesResponse

class QueryAnalyzer:
    """Service for analyzing queries and generating follow-up questions"""
    
    def __init__(self, model_provider: ModelProvider, model_name: str):
        """Initialize the query analyzer
        
        Args:
            model_provider: The model provider to use for analysis
            model_name: The name of the model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
    
    async def determine_research_parameters(self, query: str) -> BreadthDepthResponse:
        """Determine appropriate research breadth and depth based on query complexity.
        
        Args:
            query: The user's research query
            
        Returns:
            BreadthDepthResponse with breadth (1-10), depth (1-5), and explanation
        """
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

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": "You are a research planning assistant that determines appropriate research parameters."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model provider
            print("Determining research breadth and depth...")
            
            response = await self.model_provider.async_completion(
                model=self.model_name,
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
        user_prompt = f"""
        Given the following query from the user, ask some follow up questions to clarify the research direction.

        Return a maximum of {max_questions} questions, but feel free to return less if the original query is clear: <query>{query}</query>
        
        Your response should be in JSON format as follows:
        {{
          "follow_up_queries": [
            "Question 1?",
            "Question 2?",
            "Question 3?"
          ]
        }}
        """

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": "You are a research assistant that generates follow-up questions to clarify research direction."},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model provider
            print("Generating follow-up questions...")
            
            response = await self.model_provider.async_completion(
                model=self.model_name,
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