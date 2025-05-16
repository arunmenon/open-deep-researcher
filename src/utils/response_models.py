from typing import List
from pydantic import BaseModel

# Pydantic models for schema validation
class BreadthDepthResponse(BaseModel):
    breadth: int
    depth: int
    explanation: str

class FollowUpQueriesResponse(BaseModel):
    follow_up_queries: List[str]

class QueriesResponse(BaseModel):
    queries: List[str]

class QuerySimilarityResponse(BaseModel):
    are_similar: bool
    
class ProcessResultResponse(BaseModel):
    learnings: List[str]
    follow_up_questions: List[str]