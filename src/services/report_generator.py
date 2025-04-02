from typing import Dict, List, Any

from ..models.provider import ModelProvider

class ReportGenerator:
    """Service for generating research reports"""
    
    def __init__(self, model_provider: ModelProvider, model_name: str):
        """Initialize the report generator
        
        Args:
            model_provider: The model provider to use
            model_name: The name of the model to use
        """
        self.model_provider = model_provider
        self.model_name = model_name
    
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

        # Define messages for the model provider
        messages = [
            {"role": "system", "content": "You are a creative research analyst that synthesizes findings into engaging reports."},
            {"role": "user", "content": user_prompt}
        ]

        print("Generating final report...\n")

        try:
            # Make the API call through our model provider
            response = await self.model_provider.async_completion(
                model=self.model_name,
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