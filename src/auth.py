import hashlib
import sys

from fastapi import Header, HTTPException

from src.database.db import get_db_cursor
from src.exception import CustomException
from src.logger import logging


async def get_current_user(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    FastAPI dependency: verifies the X-API-Key header against Postgres
    and returns the matching user_id. Raises 401 if missing/invalid/inactive.

    Usage in a route:
        async def some_route(user_id: str = Depends(get_current_user)):
            ...
    """
    try:
        key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

        with get_db_cursor(commit=False) as cur:
            cur.execute(
                "SELECT id FROM users WHERE api_key_hash = %s AND is_active = true",
                (key_hash,),
            )
            row = cur.fetchone()

        if row is None:
            logging.warning("Auth failed: invalid or inactive API key")
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        return str(row["id"])
    except HTTPException:
        raise
    except Exception as e:
        raise CustomException(e, sys)