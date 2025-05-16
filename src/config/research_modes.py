from typing import Dict, Any, NamedTuple

class ResearchModeConfig(NamedTuple):
    """Configuration for a research mode"""
    name: str
    max_queries: int  # Maximum number of concurrent queries
    max_depth: int    # Maximum recursion depth
    max_follow_up_questions: int  # Maximum follow-up questions per query
    allow_recursion: bool  # Whether to allow recursive research
    breadth_reduction_factor: float  # How much to reduce breadth at deeper levels
    temperature: float  # Temperature for query generation
    report_temperature: float  # Temperature for report generation
    description: str  # Human-readable description

class ResearchModes:
    """Container for all research mode configurations"""
    
    FAST = ResearchModeConfig(
        name="fast",
        max_queries=3,
        max_depth=1,  # No recursion
        max_follow_up_questions=2,
        allow_recursion=False,
        breadth_reduction_factor=1.0,  # No reduction
        temperature=0.7,
        report_temperature=0.8,
        description="Quick, surface-level research with limited scope"
    )
    
    BALANCED = ResearchModeConfig(
        name="balanced",
        max_queries=7,
        max_depth=1,  # No recursion
        max_follow_up_questions=3,
        allow_recursion=False,
        breadth_reduction_factor=1.0,  # No reduction
        temperature=0.8,
        report_temperature=0.9,
        description="Moderate depth and breadth for general research needs"
    )
    
    COMPREHENSIVE = ResearchModeConfig(
        name="comprehensive",
        max_queries=5,  # Lower than balanced because of recursion
        max_depth=3,  # Allow 3 levels of recursion
        max_follow_up_questions=5,
        allow_recursion=True,
        breadth_reduction_factor=0.5,  # Reduce breadth by half at each level
        temperature=1.0,  # More creative for comprehensive research
        report_temperature=1.0,
        description="Exhaustive in-depth research with recursive exploration"
    )
    
    @classmethod
    def get_mode_config(cls, mode_name: str) -> ResearchModeConfig:
        """Get the configuration for a specified mode
        
        Args:
            mode_name: Name of the mode ('fast', 'balanced', or 'comprehensive')
            
        Returns:
            ResearchModeConfig for the specified mode
            
        Raises:
            ValueError: If an invalid mode name is provided
        """
        mode_map = {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "comprehensive": cls.COMPREHENSIVE
        }
        
        if mode_name.lower() not in mode_map:
            raise ValueError(f"Invalid mode: {mode_name}. Valid modes are: {', '.join(mode_map.keys())}")
        
        return mode_map[mode_name.lower()]