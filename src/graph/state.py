from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain_core.documents import Document #type:ignore
from langchain_core.messages import BaseMessage #type:ignore
from langgraph.graph.message import add_messages #type:ignore


class RAGState(TypedDict, total=False):
    # Request inputs
    question: str
    collection_names: List[str]
    session_id: str
    user_id: str
    message_attachments: Optional[List[dict]]

    # Internal routing & context
    is_summary: bool
    chat_history: List[dict]
    docs: List[Document]

    # Output & Persistence
    raw_answer: str
    final_answer: str
    citations: str

    # Error tracking
    error_code: Optional[str]
    error_message: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    collection_names: Optional[List[str]]
    session_id: str
    user_id: str
