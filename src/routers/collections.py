from fastapi import APIRouter, HTTPException

from src.components.vector_store import VectorStore
from src.logger import logging
from src.utils.helpers import build_error_response

router = APIRouter(tags=["Collections"])


@router.get("/collections")
async def list_collections():
    try:
        collections = await VectorStore().alist_collections()
        return {"collections": collections}
    except Exception as exc:
        logging.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    try:
        logging.info("Deleting collection: %s", collection_name)
        deleted = await VectorStore(collection_name=collection_name).adelete_collection()
        if not deleted:
            return build_error_response(
                status_code=404,
                error_code="collection_not_found",
                message="Collection not found.",
                extra={"missing_collections": [collection_name]},
            )
        return {"success": True, "deleted_collection": collection_name}
    except Exception as exc:
        logging.exception("Failed to delete collection")
        raise HTTPException(status_code=500, detail=str(exc))
