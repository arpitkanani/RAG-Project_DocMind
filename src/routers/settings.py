from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.auth import GUEST_USER_ID, _ensure_guest_user, get_current_user
from src.components.embedder import Embedder
from src.components.retriever import _qdrant_db_cache
from src.database.db import get_db_cursor
from src.logger import logging

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class SetEmbeddingProviderRequest(BaseModel):
    provider: str


@router.get("/embedding")
async def get_embedding_settings(user_id: str = Depends(get_current_user)):
    """Returns the current active embedding provider and metadata about all available options."""
    try:
        active = Embedder.get_active_provider(user_id=user_id)
        azure_specs = Embedder.get_model_specs("azure")
        bge_specs = Embedder.get_model_specs("bge")
        bge_available = Embedder.is_bge_available()

        return {
            "active_provider": active,
            "providers": {
                "azure": {
                    "id": "azure",
                    "name": f"Azure OpenAI ({azure_specs['model']})",
                    "model": azure_specs["model"],
                    "dimension": azure_specs["dimension"],
                    "available": True,
                    "description": "Hosted cloud embeddings with 1536-dimensional representations.",
                },
                "bge": {
                    "id": "bge",
                    "name": f"HuggingFace ({bge_specs['model']})",
                    "model": bge_specs["model"],
                    "dimension": bge_specs["dimension"],
                    "available": bge_available,
                    "description": "Local CPU embeddings with 384-dimensional representations (loaded once in memory).",
                },
            },
        }
    except Exception as e:
        logging.exception("Failed to get embedding settings")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding")
async def update_embedding_settings(body: SetEmbeddingProviderRequest, user_id: str = Depends(get_current_user)):
    """Updates the user's active embedding provider selection."""
    provider = body.provider.strip().lower()
    if provider not in ("azure", "bge"):
        raise HTTPException(status_code=400, detail="Invalid provider. Must be 'azure' or 'bge'.")

    if provider == "bge" and not Embedder.is_bge_available():
        raise HTTPException(status_code=400, detail="Local BGE model is not available in this environment.")

    try:
        if user_id == GUEST_USER_ID:
            _ensure_guest_user()

        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO user_settings (user_id, active_embedding_provider, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (user_id) DO UPDATE
                    SET active_embedding_provider = EXCLUDED.active_embedding_provider,
                        updated_at = now();
                """,
                (user_id, provider),
            )

        # Clear retriever vector store cache so old embedding model instances are discarded immediately
        _qdrant_db_cache.clear()
        logging.info("Cleared in-memory retriever vector store cache after embedding provider switch")

        specs = Embedder.get_model_specs(provider)
        logging.info("Updated active embedding provider for user %s to: %s (%s, %dd)", user_id, provider, specs["model"], specs["dimension"])

        return {
            "success": True,
            "active_provider": provider,
            "model": specs["model"],
            "dimension": specs["dimension"],
            "message": f"Embedding model changed to {provider} ({specs['model']}). Existing documents indexed with another model will require re-indexing or switching back.",
        }
    except Exception as e:
        logging.exception("Failed to update embedding settings")
        raise HTTPException(status_code=500, detail=str(e))
