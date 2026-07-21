"""
schemas.py
----------
Pydantic models defining the FastAPI request/response contracts.
Keeping these separate from main.py makes the API self-documenting
in the auto-generated OpenAPI docs (/docs).
"""

from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    query: str


class SourceChunk(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int


class ResetRequest(BaseModel):
    session_id: str
