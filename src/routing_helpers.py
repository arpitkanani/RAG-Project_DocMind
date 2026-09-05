from pathlib import Path
import re
from typing import List, Optional
import uuid

from fastapi.responses import JSONResponse

from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.exception import KnowledgeBaseEmptyError
from src.logger import logging
from src.schemas import QueryRequest

TEMPLATES_DIR = Path("templates")
PARTIALS_DIR = TEMPLATES_DIR / "partials"


def read_template(template_name: str) -> str:
    template_path = TEMPLATES_DIR / template_name
    with template_path.open("r", encoding="utf-8") as file_obj:
        content = file_obj.read()

    include_pattern = re.compile(r"<!--\s*INCLUDE:\s*([a-zA-Z0-9_\-]+)\s*-->")

    def _replace_include(match: re.Match) -> str:
        partial_name = match.group(1).strip()
        partial_path = PARTIALS_DIR / f"{partial_name}.html"
        if partial_path.is_file():
            with partial_path.open("r", encoding="utf-8") as pf:
                return pf.read()
        return match.group(0)

    return include_pattern.sub(_replace_include, content)


def build_error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    extra: Optional[dict] = None,
) -> JSONResponse:
    payload = {"success": False, "error_code": error_code, "message": message}
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def normalize_collection_scope(request: QueryRequest) -> List[str] | None:
    if request.collection_names:
        return [name for name in request.collection_names if name]
    if request.collection_name:
        return [request.collection_name]
    return None


def build_collection_name(source_name: str, prefix: str = "doc") -> str:
    stem = Path(source_name or prefix).stem.lower()
    safe_stem = re.sub(r"[^a-z0-9_-]+", "_", stem).strip("_") or prefix
    return f"{prefix}_{safe_stem}_{uuid.uuid4().hex[:8]}"


async def aresolve_session_scope(
    session_id: str, requested_scope: Optional[List[str]], user_id: str
) -> List[str]:
    """Asynchronously resolves session scope, enforcing collection ownership.

    When the client explicitly requests specific collections, each collection is
    verified to actually belong to this user's session before being returned.
    This prevents one user from querying another user's collection by crafting a
    ``collection_names`` payload.
    """
    available_collections = await VectorStore().alist_collections()
    memory = MemoryManager(session_id=session_id, user_id=user_id)

    if requested_scope:
        # Ownership check: filter down to only the collections that are
        # both available in Qdrant AND attached to this user's session.
        await memory.acleanup_attachments(available_collections)
        user_collections = set(await memory.aget_attachment_collections())
        owned = [c for c in requested_scope if c in user_collections]
        if not owned:
            from src.exception import CollectionNotFoundError
            raise CollectionNotFoundError(requested_scope)
        return owned

    # No explicit scope: use whatever is attached to this session.
    await memory.acleanup_attachments(available_collections)
    attachments = await memory.aget_attachment_collections()
    if attachments:
        return attachments

    raise KnowledgeBaseEmptyError(
        "Please upload a document or add a YouTube video first."
    )


def resolve_session_scope(
    session_id: str, requested_scope: Optional[List[str]], user_id: str
) -> List[str]:
    """Synchronous version of aresolve_session_scope with the same ownership rules."""
    available_collections = VectorStore().list_collections()
    memory = MemoryManager(session_id=session_id, user_id=user_id)

    if requested_scope:
        memory.cleanup_attachments(available_collections)
        user_collections = set(memory.get_attachment_collections())
        owned = [c for c in requested_scope if c in user_collections]
        if not owned:
            from src.exception import CollectionNotFoundError
            raise CollectionNotFoundError(requested_scope)
        return owned

    memory.cleanup_attachments(available_collections)
    attachments = memory.get_attachment_collections()
    if attachments:
        return attachments

    raise KnowledgeBaseEmptyError(
        "Please upload a document or add a YouTube video first."
    )


def delete_collections(collection_names: List[str]) -> List[str]:
    deleted: List[str] = []
    seen = set()

    for collection_name in collection_names:
        if not collection_name or collection_name in seen:
            continue
        seen.add(collection_name)

        try:
            if VectorStore(collection_name=collection_name).delete_collection():
                deleted.append(collection_name)
        except Exception:
            logging.exception("Failed to delete collection during cleanup: %s", collection_name)

    return deleted


async def adelete_collections(collection_names: List[str]) -> List[str]:
    deleted: List[str] = []
    seen = set()

    for collection_name in collection_names:
        if not collection_name or collection_name in seen:
            continue
        seen.add(collection_name)

        try:
            if await VectorStore(collection_name=collection_name).adelete_collection():
                deleted.append(collection_name)
        except Exception:
            logging.exception("Failed to delete collection during cleanup: %s", collection_name)

    return deleted


def get_available_collections() -> List[str]:
    try:
        return VectorStore().list_collections()
    except Exception:
        logging.exception("Failed to list available collections")
        return []


async def aget_available_collections() -> List[str]:
    try:
        return await VectorStore().alist_collections()
    except Exception:
        logging.exception("Failed to list available collections")
        return []


def create_session_id() -> str:
    session_id = str(uuid.uuid4())[:8]
    logging.info("New session: %s", session_id)
    return session_id
