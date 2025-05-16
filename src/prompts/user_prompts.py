"""
User prompts used across the application.

These define the specific instructions/queries sent to the model for each task.
"""

def research_parameters_prompt(query: str) -> str:
    """Generate prompt for determining research parameters."""
    return f"""
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

def follow_up_questions_prompt(query: str, max_questions: int) -> str:
    """Generate prompt for creating follow-up questions."""
    return f"""
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

def generate_queries_prompt(query: str, num_queries: int, previous_queries_text: str, learnings_prompt: str) -> str:
    """Generate prompt for creating search queries."""
    return f"""
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
    {learnings_prompt}
    """

def query_similarity_prompt(query1: str, query2: str) -> str:
    """Generate prompt for checking query similarity."""
    return f"""
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

def search_prompt(query: str) -> str:
    """Generate prompt for performing a search."""
    return f"Please research and provide detailed information about: {query}"

def process_result_prompt(query: str, result: str, num_learnings: int, num_follow_up_questions: int) -> str:
    """Generate prompt for processing search results."""
    return f"""
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

def generate_report_prompt(query: str, learnings_text: str, sources_text: str) -> str:
    """Generate prompt for creating the final report."""
    return f"""
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