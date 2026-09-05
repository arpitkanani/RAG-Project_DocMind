from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.documents import Document


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

    # CRAG Refine Fields
    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    # Output & Persistence
    raw_answer: str
    final_answer: str
    citations: str

    # Error tracking
    error_code: Optional[str]
    error_message: Optional[str]
