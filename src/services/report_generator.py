from typing import Dict, List, Any

from ..models.strategy import ModelStrategy
from ..prompts import system_prompts, user_prompts

class ReportGenerator:
    """Service for generating research reports"""
    
    def __init__(self, model_strategy: ModelStrategy):
        """Initialize the report generator
        
        Args:
            model_strategy: The model strategy to use
        """
        self.model_strategy = model_strategy
    
    async def generate_final_report(self, query: str, learnings: List[str], visited_urls: Dict[int, Dict], temperature: float = 0.9) -> str:
        """Generate a final comprehensive research report based on collected learnings
        
        Args:
            query: The original research query
            learnings: List of extracted learnings from research
            visited_urls: Dictionary of URLs used as sources
            temperature: Temperature for report generation (creativity)
            
        Returns:
            Formatted report text with sources
        """
        # Format sources and learnings for the prompt
        sources_text = "\n".join([
            f"- {data['title']}: {data['link']}"
            for data in visited_urls.values()
        ])
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])

        # Get the prompt from the externalized prompts module
        user_prompt = user_prompts.generate_report_prompt(
            query, learnings_text, sources_text)

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": system_prompts.REPORT_GENERATOR},
            {"role": "user", "content": user_prompt}
        ]

        print("Generating final report...\n")

        try:
            # Make the API call through our model strategy
            response = await self.model_strategy.execute_completion(
                messages=messages,
                temperature=temperature,  # Increased for more creativity
                max_tokens=8192
            )
            
            # Extract the response text
            if hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                formatted_text = response.choices[0].message.content
            else:
                # Fallback to string conversion for any other response type
                formatted_text = str(response)
                    
            # If no response, use a default message
            if not formatted_text:
                formatted_text = f"Error: No content generated for report on {query}."
                
            print(f"Final report content (first 500 chars):\n{formatted_text[:500]}...")
            
        except Exception as e:
            print(f"Error generating final report: {e}")
            
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