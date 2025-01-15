# app/models.py
from operator import add
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
class RouteQuery(BaseModel):
    """Route a user query to the most relevant node."""
    datasource: Literal["retrieve", "conversation"] = Field(
        ...,
        description="Given a user question choose to route it to retrieve or conversation.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generation.
        documents: List of retrieved documents.
        query_rewritten_num: Number of times the query has been rewritten.
    """
    question: str
    generation: str
    documents: List[str]
    query_rewritten_num: int
    final_output: Annotated[List[str], add]