from typing import List, Dict, Set, Any
import re
from ..models.provider import ModelProvider
from ..utils.response_parser import ResponseParser
from ..utils.response_models import QueriesResponse, QuerySimilarityResponse
from ..prompts import system_prompts, user_prompts

class QueryGenerator:
    """Service for generating search queries"""
    
    def __init__(self, model_provider: ModelProvider, model_name: str):
        """Initialize the query generator
        
        Args:
            model_provider: The model provider to use
            model_name: The name of the model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.query_history = set()
    
    async def generate_queries(
            self,
            query: str,
            num_queries: int = 3,
            learnings: List[str] = [],
            previous_queries: Set[str] = None,
            temperature: float = 1.0
    ) -> List[str]:
        """Generate search queries based on the input query
        
        Args:
            query: The main research query
            num_queries: Maximum number of queries to generate
            learnings: Optional list of previous learnings to improve query generation
            previous_queries: Set of previously generated queries to avoid duplicates
            temperature: Temperature for query generation (creativity)
            
        Returns:
            List of generated search queries
        """
        # Format previous queries for the prompt
        previous_queries_text = ""
        if previous_queries:
            previous_queries_text = "\n\nPreviously asked queries (avoid generating similar ones):\n" + \
                "\n".join([f"- {q}" for q in previous_queries])

        # Format learnings for the prompt
        learnings_prompt = "" if not learnings else "\n\nHere are some learnings from previous research, use them to generate more specific queries: " + \
            "\n".join([f"- {learning}" for learning in learnings])

        # Get the prompt from the externalized prompts module
        user_prompt = user_prompts.generate_queries_prompt(
            query, num_queries, previous_queries_text, learnings_prompt)

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": system_prompts.QUERY_GENERATOR},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model provider
            print("Generating search queries...")
            
            response = await self.model_provider.async_completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )
            
            output_text = response.choices[0].message.content
            
            # Parse the response using ResponseParser
            try:
                queries_response = ResponseParser.parse_json_response(output_text, QueriesResponse)
                generated_queries = queries_response.queries
                
                # Filter out queries that are too similar to ones we've already asked
                unique_queries = []
                for query in generated_queries:
                    is_similar = False
                    for history_query in self.query_history:
                        if await self._are_queries_similar(query, history_query):
                            is_similar = True
                            break
                    
                    if not is_similar:
                        unique_queries.append(query)
                        self.query_history.add(query)
                
                return unique_queries[:num_queries]
                
            except ValueError as e:
                print(f"Could not parse JSON for query generation: {e}")
                # Generate some default queries if extraction fails
                return [
                    f"{query} research papers",
                    f"{query} latest developments",
                    f"{query} technical details"
                ][:num_queries]
            
        except Exception as e:
            print(f"Error generating queries: {e}")
            
            # Generate some default queries
            print("Using default queries.")
            return [
                f"{query} research papers",
                f"{query} latest developments",
                f"{query} technical details"
            ][:num_queries]
    
    async def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """Helper method to check if two queries are semantically similar"""
        # Get the prompt from the externalized prompts module
        user_prompt = user_prompts.query_similarity_prompt(query1, query2)

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": system_prompts.QUERY_SIMILARITY_CHECKER},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Make the API call through our model provider
            response = await self.model_provider.async_completion(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=1024
            )

            # Extract and parse the response
            output_text = response.choices[0].message.content
            similarity_response = ResponseParser.parse_json_response(output_text, QuerySimilarityResponse)
            return similarity_response.are_similar
            
        except Exception as e:
            print(f"Error checking query similarity: {e}")
            # In case of error, assume queries are different to avoid missing potentially unique results
            return False