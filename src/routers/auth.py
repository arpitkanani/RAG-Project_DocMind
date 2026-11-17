import hashlib
import os
import re
import secrets
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import anyio
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.auth import (
    create_access_token,
    decode_access_token,
    authenticate_admin,
    authenticate_customer,
    register_admin,
    register_customer,
    get_current_user_optional,
    SESSION_MAX_DAYS,
)
from src.database.db import get_db_cursor
from src.utils.perf import perf_step
from src.logger import logging

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class VerifyKeyRequest(BaseModel):
    api_key: str


class CreateKeyRequest(BaseModel):
    username: Optional[str] = None
    force_rotate: Optional[bool] = False


class RegisterAdminRequest(BaseModel):
    username: str
    email: str
    password: str


class RegisterCustomerRequest(BaseModel):
    username: str
    password: str


class LoginAdminRequest(BaseModel):
    email: str
    password: str


class LoginCustomerRequest(BaseModel):
    username: str
    password: str


def _get_candidate_usernames(username_clean: str) -> list[str]:
    u_lower = username_clean.lower()
    candidates = [u_lower]
    if u_lower == "test":
        candidates.append("test_user")
    elif u_lower == "test_user":
        candidates.append("test")
    elif u_lower.endswith("_user"):
        candidates.append(u_lower[:-5])
    else:
        candidates.append(f"{u_lower}_user")
    return list(dict.fromkeys(candidates))


@router.post("/create-key")
async def create_key(body: Optional[CreateKeyRequest] = None):
    """
    Generates a dk_live_... API key directly from the web application, mirroring seed_user.py.
    - If username does not exist: creates a new user in PostgreSQL and issues the key.
    - If username exists and force_rotate=False: notifies the client that the account exists
      with previous chat history.
    - If username exists and force_rotate=True: re-issues the key for that existing user_id,
      keeping all previous sessions, messages, and uploaded files 100% intact!
    """
    raw_username = (body.username if body and body.username else "").strip()
    username_clean = re.sub(r"[^a-zA-Z0-9_\-]", "", raw_username) if raw_username else ""
    if not username_clean or len(username_clean) < 3:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "A valid username of at least 3 characters (letters, numbers, underscores, dashes) is required.",
            },
        )

    force_rotate = bool(body.force_rotate) if body else False
    email_clean = f"{username_clean.lower()}@docuvortex.local"
    candidates = _get_candidate_usernames(username_clean)
    candidate_emails = [f"{c}@docuvortex.local" for c in candidates]

    try:
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                """
                SELECT id, username, email FROM users
                WHERE lower(username) = ANY(%s) OR lower(email) = ANY(%s)
                ORDER BY created_at ASC
                LIMIT 1;
                """,
                (candidates, candidate_emails),
            )
            existing = cur.fetchone()

            if existing and not force_rotate:
                # Account already exists; inform frontend with latest session info so user can rotate
                existing_user_id = str(existing["id"])
                last_session_id = None
                session_count = 0
                try:
                    cur.execute(
                        """
                        SELECT session_id FROM sessions
                        WHERE user_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1;
                        """,
                        (existing_user_id,),
                    )
                    s_row = cur.fetchone()
                    if s_row:
                        last_session_id = s_row["session_id"]
                    cur.execute(
                        "SELECT count(*) as cnt FROM sessions WHERE user_id = %s;",
                        (existing_user_id,),
                    )
                    c_row = cur.fetchone()
                    if c_row:
                        session_count = c_row.get("cnt", 0)
                except Exception as sess_lookup_err:
                    logging.debug("Session count lookup note: %s", sess_lookup_err)

                return {
                    "success": True,
                    "exists": True,
                    "username": existing["username"] or username_clean,
                    "last_session_id": last_session_id,
                    "session_count": session_count,
                    "message": f"An account with username '{existing['username'] or username_clean}' already exists.",
                }

            raw_api_key = f"dk_live_{secrets.token_hex(24)}"
            api_key_hash = hashlib.sha256(raw_api_key.encode()).hexdigest()

            if existing:
                user_id = str(existing["id"])
                cur.execute(
                    """
                    UPDATE users
                    SET api_key_hash = %s,
                        is_active = true,
                        is_verified = true
                    WHERE id = %s;
                    """,
                    (api_key_hash, user_id),
                )
                is_existing = True
                logging.info("Rotated API key for existing user: %s (id: %s)", username_clean, user_id)
            else:
                user_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO users (id, username, email, api_key_hash, name, is_verified, is_active)
                    VALUES (%s, %s, %s, %s, %s, true, true);
                    """,
                    (user_id, username_clean, email_clean, api_key_hash, username_clean),
                )
                is_existing = False
                logging.info("Created new user via web: %s (id: %s)", username_clean, user_id)

            # Associate any orphaned guest chats or duplicate account chats for this username to this user_id
            try:
                cur.execute(
                    """
                    UPDATE sessions
                    SET user_id = %s
                    WHERE user_id = '00000000-0000-0000-0000-000000000001'
                       OR user_id IN (
                           SELECT id FROM users
                           WHERE (lower(username) = ANY(%s) OR lower(email) = ANY(%s))
                             AND id != %s
                       );
                    """,
                    (user_id, candidates, candidate_emails, user_id),
                )
            except Exception as mig_err:
                logging.debug("Session association note: %s", mig_err)

            last_session_id = None
            try:
                cur.execute(
                    """
                    SELECT session_id
                    FROM sessions
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1;
                    """,
                    (user_id,),
                )
                last_s_row = cur.fetchone()
                if last_s_row:
                    last_session_id = last_s_row["session_id"]
            except Exception as sess_err:
                logging.debug("create-key last session note: %s", sess_err)

        return {
            "success": True,
            "exists": False,
            "api_key": raw_api_key,
            "username": username_clean,
            "user_id": user_id,
            "is_existing": is_existing,
            "last_session_id": last_session_id,
            "message": "API key rotated successfully." if is_existing else "API key generated successfully.",
        }
    except Exception as e:
        logging.exception("create-key error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create API key.")


@router.post("/verify-key")
async def verify_key(body: VerifyKeyRequest, response: Response):
    """Validates an API key generated by seed_user.py or /api/auth/create-key."""
    key = body.api_key.strip() if body.api_key else ""
    if not key:
        return {"valid": False, "message": "Please enter an API key."}
    if key == "guest":
        return {"valid": False, "message": "Guest mode is disabled. Please provide a valid DocuVortex API key."}

    key_hash = hashlib.sha256(key.encode()).hexdigest()
    try:
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                "SELECT id, username, email FROM users WHERE api_key_hash = %s LIMIT 1",
                (key_hash,),
            )
            row = cur.fetchone()
            if row:
                user_id = str(row["id"])
                username = row.get("username") or "User"
                email = row.get("email") or ""

                # Migrate any prior sessions created under guest or username aliases to this user so history is preserved
                try:
                    user_candidates = _get_candidate_usernames(username)
                    user_candidate_emails = [f"{c}@docuvortex.local" for c in user_candidates]
                    cur.execute(
                        """
                        UPDATE sessions
                        SET user_id = %s
                        WHERE user_id = '00000000-0000-0000-0000-000000000001'
                           OR user_id IN (
                               SELECT id FROM users
                               WHERE (lower(username) = ANY(%s) OR lower(email) = ANY(%s))
                                 AND id != %s
                           );
                        """,
                        (user_id, user_candidates, user_candidate_emails, user_id),
                    )
                except Exception as mig_err:
                    logging.debug("verify-key session migration note: %s", mig_err)

                last_session_id = None
                try:
                    cur.execute(
                        """
                        SELECT session_id
                        FROM sessions
                        WHERE user_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (user_id,),
                    )
                    last_row = cur.fetchone()
                    if last_row:
                        last_session_id = last_row["session_id"]
                except Exception as sess_err:
                    logging.debug("verify-key session fetch note: %s", sess_err)

                token = create_access_token(
                    user_id=user_id,
                    session_id=str(uuid.uuid4()),
                    email=email,
                    username=username,
                )
                is_dev = os.getenv("ENV", "dev").lower() in ("dev", "development")
                response.set_cookie(
                    key="access_token",
                    value=token,
                    httponly=True,
                    samesite="lax",
                    secure=not is_dev,
                    max_age=SESSION_MAX_DAYS * 86400,
                    path="/",
                )

                return {
                    "valid": True,
                    "user": {
                        "id": user_id,
                        "username": username,
                        "email": email,
                    },
                    "last_session_id": last_session_id,
                }
    except Exception as e:
        logging.warning("verify-key error: %s", e)

    return {
        "valid": False,
        "message": "Invalid API key. Please check the key generated with seed_user.py or created on landing page.",
    }


@router.get("/session-status")
async def session_status(request: Request):
    user_id = None
    username = None
    email = None
    last_session_id = None

    # 1. Check API Key header or query param (Primary authentication via seed_user.py or /create-key)
    api_key = (
        request.headers.get("X-API-Key")
        or request.query_params.get("api_key")
        or ""
    ).strip()

    if api_key and api_key != "guest":
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        try:
            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    "SELECT id, username, email FROM users WHERE api_key_hash = %s LIMIT 1",
                    (key_hash,),
                )
                row = cur.fetchone()
                if row:
                    user_id = str(row["id"])
                    username = row.get("username") or "User"
                    email = row.get("email") or ""
        except Exception as e:
            logging.debug("session-status API key check note: %s", e)

    # 2. Check Cookie fallback
    if not user_id:
        token = request.cookies.get("access_token")
        if token:
            payload = decode_access_token(token)
            if payload and "sub" in payload:
                user_id = payload["sub"]
                username = payload.get("username") or (payload.get("email", "User").split("@")[0] if payload.get("email") else "User")
                email = payload.get("email", "")

    if not user_id:
        return {"authenticated": False}

    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                "SELECT username, email FROM users WHERE id = %s LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            if row:
                username = row.get("username") or username
                email = row.get("email") or email

            # Find the user's most recently active chat session using indexed lookup
            cur.execute(
                """
                SELECT s.session_id
                FROM sessions s
                WHERE s.user_id = %s
                ORDER BY s.updated_at DESC
                LIMIT 1
                """,
                (user_id,),
            )
            last_row = cur.fetchone()
            if last_row:
                last_session_id = last_row["session_id"]
    except Exception as e:
        logging.debug("session-status DB check note: %s", e)

    return {
        "authenticated": True,
        "user": {
            "id": user_id,
            "username": username,
            "email": email,
        },
        "last_session_id": last_session_id,
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("access_token")
    if token:
        from src.auth import invalidate_token_cache
        invalidate_token_cache(token)
        payload = decode_access_token(token)
        if payload and payload.get("session_id"):
            try:
                with get_db_cursor(commit=True) as cur:
                    cur.execute("DELETE FROM user_sessions WHERE id = %s", (payload["session_id"],))
            except Exception as e:
                logging.debug("Logout session cleanup note: %s", e)

    response.delete_cookie(key="access_token", path="/", httponly=True, samesite="lax")
    response.set_cookie(
        key="access_token",
        value="",
        max_age=0,
        expires="Thu, 01 Jan 1970 00:00:00 GMT",
        path="/",
        httponly=True,
        samesite="lax",
    )
    return {"success": True, "message": "Logged out successfully", "redirect": "/"}


@router.post("/register-admin")
async def api_register_admin(body: RegisterAdminRequest, response: Response):
    try:
        data = register_admin(body.username, body.email, body.password)
        with perf_step("cookie_set"):
            response.set_cookie(
                key="access_token",
                value=data["token"],
                max_age=SESSION_MAX_DAYS * 86400,
                httponly=True,
                samesite="lax",
                path="/",
            )
        return {
            "success": True,
            "message": "Admin registered successfully",
            "user": {
                "id": data["user_id"],
                "username": data["username"],
                "email": data["email"],
                "role": "admin",
            },
            "token": data["token"],
            "api_key": data.get("api_key"),
        }
    except ValueError as e:
        err_msg = str(e)
        status_code = 409 if "already exists" in err_msg.lower() else 400
        raise HTTPException(status_code=status_code, detail=err_msg)
    except Exception as e:
        logging.exception("Failed to register admin")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


@router.post("/register-customer")
async def api_register_customer(body: RegisterCustomerRequest, response: Response):
    try:
        data = register_customer(body.username, body.password)
        with perf_step("cookie_set"):
            response.set_cookie(
                key="access_token",
                value=data["token"],
                max_age=SESSION_MAX_DAYS * 86400,
                httponly=True,
                samesite="lax",
                path="/",
            )
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "id": data["user_id"],
                "username": data["username"],
                "email": data["email"],
                "role": "user",
            },
            "token": data["token"],
            "api_key": data.get("api_key"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Failed to register user")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


@router.post("/login-admin")
async def api_login_admin(body: LoginAdminRequest, response: Response):
    data = authenticate_admin(body.email, body.password)
    if not data:
        raise HTTPException(status_code=401, detail="Invalid admin email or password.")
    with perf_step("cookie_set"):
        response.set_cookie(
            key="access_token",
            value=data["token"],
            max_age=SESSION_MAX_DAYS * 86400,
            httponly=True,
            samesite="lax",
            path="/",
        )
    return {
        "success": True,
        "message": "Admin authenticated successfully",
        "user": {
            "id": data["user_id"],
            "username": data["username"],
            "email": data["email"],
            "role": "admin",
        },
        "token": data["token"],
        "api_key": data.get("api_key"),
    }


@router.post("/login-customer")
async def api_login_customer(body: LoginCustomerRequest, response: Response):
    data = authenticate_customer(body.username, body.password)
    if not data:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    with perf_step("cookie_set"):
        response.set_cookie(
            key="access_token",
            value=data["token"],
            max_age=SESSION_MAX_DAYS * 86400,
            httponly=True,
            samesite="lax",
            path="/",
        )
    return {
        "success": True,
        "message": "User authenticated successfully",
        "user": {
            "id": data["user_id"],
            "username": data["username"],
            "email": data["email"],
            "role": data.get("role", "user"),
        },
        "token": data["token"],
        "api_key": data.get("api_key"),
        "is_new": data.get("is_new", False),
    }


@router.get("/me")
async def api_get_current_user(request: Request, response: Response):
    """Returns active authenticated user details for instant auto-resume."""
    user = await get_current_user_optional(request)
    if not user:
        return {"authenticated": False, "user": None}

    user_id = str(user["id"])
    role = user.get("role") or "user"
    username = user.get("username") or "User"
    email = user.get("email") or ""

    # Token stability: only refresh cookie if missing or nearing expiration (< 2 days remaining)
    existing_token = request.cookies.get("access_token")
    need_refresh = True
    if existing_token:
        payload = decode_access_token(existing_token)
        if payload and payload.get("exp"):
            now_ts = int(datetime.now(timezone.utc).timestamp())
            remaining_secs = payload["exp"] - now_ts
            if remaining_secs > 2 * 86400:
                need_refresh = False

    if need_refresh:
        from src.auth import _TOKEN_USER_CACHE, _TOKEN_USER_DICT_CACHE
        session_id = str(uuid.uuid4())
        token = create_access_token(user_id=user_id, session_id=session_id, email=email, username=username, role=role)
        now_m = time.monotonic()
        _TOKEN_USER_CACHE[token] = (now_m, user_id)
        _TOKEN_USER_DICT_CACHE[token] = (now_m, user)
        response.set_cookie(
            key="access_token",
            value=token,
            max_age=SESSION_MAX_DAYS * 86400,
            httponly=True,
            samesite="lax",
            path="/",
        )

    last_session_id = None
    workspace_key = None
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                "SELECT session_id FROM sessions WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
            if row:
                last_session_id = row["session_id"]
            
            # Maintain active API key if provided by client
            header_key = (request.headers.get("X-API-Key") or request.query_params.get("api_key") or "").strip()
            if header_key and header_key.startswith("dk_live_"):
                workspace_key = header_key
    except Exception as e:
        logging.debug("api_get_current_user session fetch note: %s", e)

    return {
        "authenticated": True,
        "user": user,
        "last_session_id": last_session_id,
        "api_key": workspace_key,
    }
