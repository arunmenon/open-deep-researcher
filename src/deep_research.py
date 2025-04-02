from typing import Callable, List, TypeVar, Any, Dict, Optional, Union, Set
import asyncio
import datetime
import json
import os
import re
import math

from dotenv import load_dotenv

# Import from our newly organized structure
from .utils.research_progress import ResearchProgress
from .models.provider import ModelProvider
from .models.runpod_provider import RunpodProvider
from .models.strategy import ModelStrategy, ModelStrategyFactory, RunPodStrategy

# Import services
from .services.query_analyzer import QueryAnalyzer
from .services.query_generator import QueryGenerator
from .services.search_service import SearchService
from .services.result_processor import ResultProcessor
from .services.report_generator import ReportGenerator

# Import configuration
from .config.research_modes import ResearchModes, ResearchModeConfig

load_dotenv()


class DeepSearch:
    def __init__(
        self, 
        mode: str = "balanced", 
        model_strategy: Optional[ModelStrategy] = None,
        strategy_type: Optional[str] = None,
        strategy_config: Optional[Dict[str, Any]] = None,
        model_provider: Optional[ModelProvider] = None,
        runpod_api_key: Optional[str] = None, 
        runpod_endpoint_id: Optional[str] = None
    ):
        """
        Initialize DeepSearch with a mode parameter:
        - "fast": Prioritizes speed (reduced breadth/depth, highest concurrency)
        - "balanced": Default balance of speed and comprehensiveness
        - "comprehensive": Maximum detail and coverage

        Args:
            mode: Research mode - fast, balanced, or comprehensive
            model_strategy: Optional ModelStrategy instance to use
            strategy_type: Optional type of strategy to create
            strategy_config: Optional configuration for strategy creation
            model_provider: Optional legacy ModelProvider instance
            runpod_api_key: Optional legacy RunPod API key
            runpod_endpoint_id: Optional legacy RunPod endpoint ID
        """
        # Set up the model strategy (preferred) or provider (legacy)
        if model_strategy:
            # Use provided strategy
            self.model_strategy = model_strategy
        elif strategy_type:
            # Create strategy from type and config
            self.model_strategy = ModelStrategyFactory.create_strategy(
                strategy_type, 
                strategy_config or {}
            )
        elif model_provider:
            # Legacy: Wrap provider in strategy
            from .models.strategy import ProviderModelStrategy
            self.model_strategy = ProviderModelStrategy(
                model_provider, 
                "provider-model"
            )
        else:
            # Legacy: Create RunPod strategy from parameters
            self.runpod_api_key = runpod_api_key or os.getenv("RUNPOD_API_KEY")
            if not self.runpod_api_key:
                raise ValueError("RunPod API key is required")
                
            self.endpoint_id = runpod_endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID", "w0oa6hyd2q40jw")
            self.model_strategy = RunPodStrategy(self.runpod_api_key, self.endpoint_id)
        
        # Get the research mode configuration
        self.mode_config = ResearchModes.get_mode_config(mode)
        
        # Initialize services with the model strategy
        self._init_services()
        
        print(f"Configured DeepSearch with {mode} mode using {type(self.model_strategy).__name__}")
    
    def _init_services(self):
        """Initialize all service components with the model strategy"""
        self.query_analyzer = QueryAnalyzer(self.model_strategy)
        self.query_generator = QueryGenerator(self.model_strategy)
        self.search_service = SearchService(self.model_strategy)
        self.result_processor = ResultProcessor(self.model_strategy)
        self.report_generator = ReportGenerator(self.model_strategy)
        
    async def determine_research_breadth_and_depth(self, query: str):
        """
        Determine appropriate research breadth and depth based on query complexity.
        
        Args:
            query: The user's research query
            
        Returns:
            BreadthDepthResponse with breadth (1-10), depth (1-5), and explanation
        """
        return await self.query_analyzer.determine_research_parameters(query)
        
    async def generate_follow_up_questions(self, query: str, max_questions: int = 3) -> List[str]:
        """
        Generate follow-up questions to clarify research direction
        
        Args:
            query: The initial user query
            max_questions: Maximum number of questions to generate
            
        Returns:
            List of follow-up questions
        """
        return await self.query_analyzer.generate_follow_up_questions(query, max_questions)

    async def deep_research(self, query: str, breadth: int, depth: int, learnings: list[str] = [], visited_urls: dict[int, dict] = {}, parent_query: str = None):
        """
        Perform deep research on a query
        
        Args:
            query: The research query
            breadth: Breadth of the research (1-10)
            depth: Depth of the research (1-5)
            learnings: Optional list of previous learnings
            visited_urls: Dictionary of visited URLs
            parent_query: Parent query if this is a sub-query
            
        Returns:
            Dictionary with learnings and visited URLs
        """
        progress = ResearchProgress(depth, breadth)
        
        # Start the root query
        progress.start_query(query, depth, parent_query)

        # Calculate number of queries based on research mode and breadth
        num_queries = min(breadth, self.mode_config.max_queries)

        # Generate search queries
        queries = await self.query_generator.generate_queries(
            query,
            num_queries=num_queries,
            learnings=learnings,
            temperature=self.mode_config.temperature
        )

        unique_queries = list(queries)[:breadth]

        async def process_query(query_str: str, current_depth: int, parent: str = None):
            try:
                # Start this query as a sub-query of the parent
                progress.start_query(query_str, current_depth, parent)

                # Perform search
                result = await self.search_service.search(query_str)
                
                # Process results
                processed_result = await self.result_processor.process_result(
                    query=query_str,
                    result=result[0],
                    num_learnings=min(3, math.ceil(breadth / 2)),
                    num_follow_up_questions=min(2, math.ceil(breadth / 2))
                )

                # Record learnings
                for learning in processed_result["learnings"]:
                    progress.add_learning(query_str, current_depth, learning)

                # Update visited URLs
                new_urls = result[1]
                max_idx = max(visited_urls.keys()) if visited_urls else -1
                all_urls = {
                    **visited_urls,
                    **{(i + max_idx + 1): url_data for i, url_data in new_urls.items()}
                }

                # Only go deeper if recursion is allowed and depth > 1
                if self.mode_config.allow_recursion and current_depth > 1:
                    # Reduced breadth for deeper levels
                    new_breadth = min(2, math.ceil(breadth * self.mode_config.breadth_reduction_factor))
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

        # Deduplicate URLs
        all_urls = {}
        current_idx = 0
        seen_urls = set()
        for result in results:
            for url_data in result["visited_urls"].values():
                if url_data.get('link', '') not in seen_urls:
                    all_urls[current_idx] = url_data
                    seen_urls.add(url_data.get('link', ''))
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

    async def generate_final_report(self, query: str, learnings: list[str], visited_urls: dict[int, dict]) -> str:
        """
        Generate a final comprehensive research report based on collected learnings
        
        Args:
            query: The original research query
            learnings: List of extracted learnings from research
            visited_urls: Dictionary of URLs used as sources
            
        Returns:
            Formatted report text with sources
        """
        return await self.report_generator.generate_final_report(
            query=query,
            learnings=learnings,
            visited_urls=visited_urls,
            temperature=self.mode_config.report_temperature
        )