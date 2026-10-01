from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User question payload."""

    query: str
    collection_name: Optional[str] = None
    collection_names: Optional[List[str]] = None
    message_attachments: Optional[List[dict]] = None
    session_id: Optional[str] = "default"


class YouTubeRequest(BaseModel):
    """YouTube URL processing payload."""

    url: str
    session_id: Optional[str] = "default"
    collection_name: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str = Field(..., description="Unique 8-character session identifier")



