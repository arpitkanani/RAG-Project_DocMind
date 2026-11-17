import asyncio
import json
import os
import re
import sys
import uuid
from typing import List, Optional
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF fallback
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from business_assistant.agent import BusinessBotChatSession
from business_assistant.architect import MasterArchitectInterview
from business_assistant.prompts import generate_dynamic_prompt_with_azure
from business_assistant.scraper import scrape_single_url, scrape_multiple_urls
from src.auth import (
    get_current_admin,
    get_current_user_optional,
    decode_access_token,
)
from src.database.db import get_db_cursor
from src.logger import logging
from src.components.vector_store import VectorStore
from langchain_core.documents import Document
from src.utils.helpers import read_template

router = APIRouter(tags=["Business Assistant"])

# Store in-memory active interviews by interview_id
_active_interviews: dict[str, MasterArchitectInterview] = {}
# Active bot existence cache to avoid repeated DB roundtrips on customer bot page
_ACTIVE_BOT_IDS_CACHE: set[str] = set()


# ═══════════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════════

@router.get("/builder", response_class=HTMLResponse)
async def builder_page(request: Request):
    """Admin bot creation wizard page. Guarded: redirects unauthenticated users."""
    user = await get_current_user_optional(request)
    if not user:
        return RedirectResponse(url="/?action=login&role=admin", status_code=303)
    if user.get("role") != "admin":
        return RedirectResponse(url="/?action=register_admin&role=admin", status_code=303)
    try:
        return read_template("builder.html")
    except Exception as e:
        logging.error("Failed rendering builder.html: %s", e)
        raise HTTPException(status_code=500, detail="Could not load builder page")


@router.get("/admin-dashboard", response_class=HTMLResponse)
async def admin_dashboard_page(request: Request):
    """Admin control center page. Guarded: redirects non-admins."""
    user = await get_current_user_optional(request)
    if not user:
        return RedirectResponse(url="/?action=login&role=admin", status_code=303)
    if user.get("role") != "admin":
        return RedirectResponse(url="/?action=register_admin&role=admin", status_code=303)
    try:
        return read_template("admin_dashboard.html")
    except Exception as e:
        logging.error("Failed rendering admin_dashboard.html: %s", e)
        raise HTTPException(status_code=500, detail="Could not load admin dashboard")


@router.get("/bot/{bot_id}", response_class=HTMLResponse)
async def customer_bot_page(bot_id: str):
    """Customer chat page for a deployed bot instance."""
    if bot_id not in _ACTIVE_BOT_IDS_CACHE:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT bot_id FROM deployed_bots WHERE bot_id = %s LIMIT 1", (bot_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Bot not found")
        _ACTIVE_BOT_IDS_CACHE.add(bot_id)
    try:
        return read_template("business_chat.html")
    except Exception as e:
        logging.error("Failed rendering business_chat.html: %s", e)
        raise HTTPException(status_code=500, detail="Could not load chat page")


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC APIs
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/business-bot/public/list")
async def public_bot_list():
    """Lists all active deployed bots for the landing page marketplace."""
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT bot_id, business_name, persona_name, business_summary,
                       admin_id, website_url, created_at
                FROM deployed_bots
                WHERE status = 'active'
                ORDER BY created_at DESC;
                """
            )
            rows = cur.fetchall() or []
            return [
                {
                    "id": r["bot_id"],
                    "bot_id": r["bot_id"],
                    "business_name": r["business_name"],
                    "persona_name": r["persona_name"],
                    "summary": r["business_summary"] or f"Customer assistant for {r['business_name']}",
                    "business_summary": r["business_summary"] or f"Customer assistant for {r['business_name']}",
                    "admin_id": str(r["admin_id"]) if r.get("admin_id") else "",
                    "website_url": r.get("website_url") or "",
                    "created_at": r["created_at"].isoformat() if r.get("created_at") else "",
                }
                for r in rows
            ]
    except Exception as e:
        logging.exception("Failed fetching public bots list")
        return []


@router.get("/api/business-bot/bot/{bot_id}")
@router.get("/api/business-bot/public/get/{bot_id}")
async def get_bot_metadata(bot_id: str):
    """Fetches public metadata for a specific deployed bot."""
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT bot_id, business_name, persona_name, business_summary,
                   initial_greeting, booking_contact, website_url
            FROM deployed_bots
            WHERE bot_id = %s LIMIT 1;
            """,
            (bot_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Bot not found")
        data = dict(row)
        data["id"] = data["bot_id"]
        return {
            "success": True,
            "bot": data,
            **data,
        }


# ═══════════════════════════════════════════════════════════════════
#  ADMIN APIs (Guarded by get_current_admin)
# ═══════════════════════════════════════════════════════════════════

class ScrapeUrlRequest(BaseModel):
    url: Optional[str] = None
    urls: Optional[List[str]] = None


@router.post("/api/business-bot/scrape-url")
@router.post("/api/business-bot/admin/scrape-url")
async def api_scrape_url(body: ScrapeUrlRequest, admin: dict = Depends(get_current_admin)):
    """Scrapes one or multiple webpages (up to 5) without recursive link crawling."""
    urls_to_scrape = []
    if body.urls:
        urls_to_scrape.extend(body.urls)
    if body.url:
        urls_to_scrape.append(body.url)

    result = await scrape_multiple_urls(urls_to_scrape)
    return result


def _split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    clean = (text or "").strip()
    if not clean:
        return chunks
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(clean):
        c = clean[start : start + chunk_size].strip()
        if len(c) >= 20:
            chunks.append(c)
        start += step
    return chunks


@router.post("/api/business-bot/upload-brochures")
@router.post("/api/business-bot/admin/upload-brochures")
async def api_upload_brochures(
    files: List[UploadFile] = File(...),
    bot_id: Optional[str] = Form(None),
    admin: dict = Depends(get_current_admin),
):
    """
    Validates up to 3 PDFs (max 4MB each).
    Extracts text for Master Architect context and indexes into Qdrant collection biz_{bot_id}.
    """
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 brochure PDFs allowed.")

    assigned_bot_id = bot_id or f"bot_{uuid.uuid4().hex[:8]}"
    collection_name = f"biz_{assigned_bot_id}"

    extracted_chunks = []
    attachments_meta = []
    all_docs = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF. Only PDFs are supported.")

        content_bytes = await file.read()
        size_mb = len(content_bytes) / (1024 * 1024)
        if size_mb > 4.0:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' ({size_mb:.1f}MB) exceeds the 4MB limit.")

        # Extract text via PyMuPDF
        file_text = ""
        try:
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text() or ""
                file_text += page_text + "\n"
                if page_text.strip():
                    text_chunks = _split_text_into_chunks(page_text, chunk_size=800, overlap=150)
                    for c in text_chunks:
                        all_docs.append(Document(page_content=c, metadata={"source": file.filename, "page": page_num, "bot_id": assigned_bot_id}))
            extracted_chunks.append(f"=== BROCHURE: {file.filename} ===\n{file_text[:8000]}")
        except Exception as pdf_err:
            logging.warning("PyMuPDF extraction note for %s: %s", file.filename, pdf_err)

        attachments_meta.append({
            "filename": file.filename,
            "size_mb": round(size_mb, 2),
            "collection": collection_name,
        })

    # Ingest into Qdrant collection
    if all_docs:
        try:
            vs = VectorStore(collection_name=collection_name)
            vs.store_documents(all_docs)
            logging.info("Successfully indexed %d chunks for collection %s", len(all_docs), collection_name)
        except Exception as ing_err:
            logging.error("Failed indexing brochure chunks for %s: %s", collection_name, ing_err)

    combined_text = "\n\n".join(extracted_chunks)
    return {
        "success": True,
        "bot_id": assigned_bot_id,
        "attachments": attachments_meta,
        "extracted_brochures_text": combined_text[:16000],
    }


class GeneratePromptRequest(BaseModel):
    business_name: str
    persona_name: str
    website_url: Optional[str] = ""
    website_summary: Optional[str] = ""
    brochures_summary: Optional[str] = ""
    finalized_blueprint: Optional[dict] = None
    blueprint: Optional[dict] = None
    booking_contact: Optional[str] = ""


@router.post("/api/business-bot/generate-prompt")
@router.post("/api/business-bot/admin/generate-prompt")
async def api_generate_prompt(body: GeneratePromptRequest, admin: dict = Depends(get_current_admin)):
    """Invokes Azure OpenAI (gpt-5.4-nano) to synthesize the Master System Prompt."""
    try:
        blueprint_data = body.finalized_blueprint or body.blueprint or {}
        prompt = await generate_dynamic_prompt_with_azure(
            business_name=body.business_name,
            persona_name=body.persona_name,
            website_url=body.website_url or "",
            website_summary=body.website_summary or "",
            brochures_summary=body.brochures_summary or "",
            finalized_blueprint=blueprint_data,
            booking_contact=body.booking_contact or "",
        )
        return {"success": True, "prompt": prompt, "dynamic_prompt": prompt}
    except Exception as e:
        logging.exception("Failed generating dynamic prompt")
        raise HTTPException(status_code=500, detail=str(e))


class DeployBotRequest(BaseModel):
    bot_id: str
    business_name: str
    persona_name: str
    business_summary: Optional[str] = ""
    summary: Optional[str] = ""
    website_url: Optional[str] = ""
    website_data: Optional[dict] = {}
    pdf_attachments: Optional[list] = []
    finalized_blueprint: Optional[dict] = {}
    dynamic_prompt: Optional[str] = ""
    system_prompt: Optional[str] = ""
    initial_greeting: Optional[str] = ""
    booking_contact: Optional[str] = ""


@router.post("/api/business-bot/save-draft")
@router.post("/api/business-bot/admin/save-draft")
async def api_save_draft_bot(body: DeployBotRequest, admin: dict = Depends(get_current_admin)):
    """Saves or updates a draft bot configuration before public deployment."""
    admin_id = admin["id"]
    summary = body.business_summary or body.summary or f"AI Customer Assistant for {body.business_name}"
    prompt = body.dynamic_prompt or body.system_prompt or ""
    greeting = body.initial_greeting or f"Hello! Welcome to {body.business_name}. I am {body.persona_name}, your personal assistant. How can I help you today?"

    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO deployed_bots (
                bot_id, admin_id, business_name, persona_name, business_summary,
                website_url, website_data, pdf_attachments, finalized_blueprint,
                dynamic_prompt, initial_greeting, booking_contact, status, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft', now())
            ON CONFLICT (bot_id) DO UPDATE SET
                business_name = EXCLUDED.business_name,
                persona_name = EXCLUDED.persona_name,
                business_summary = EXCLUDED.business_summary,
                website_url = EXCLUDED.website_url,
                website_data = EXCLUDED.website_data,
                pdf_attachments = EXCLUDED.pdf_attachments,
                finalized_blueprint = EXCLUDED.finalized_blueprint,
                dynamic_prompt = EXCLUDED.dynamic_prompt,
                initial_greeting = EXCLUDED.initial_greeting,
                booking_contact = EXCLUDED.booking_contact,
                updated_at = now();
            """,
            (
                body.bot_id,
                admin_id,
                body.business_name,
                body.persona_name,
                summary,
                body.website_url,
                json.dumps(body.website_data or {}),
                json.dumps(body.pdf_attachments or []),
                json.dumps(body.finalized_blueprint or {}),
                prompt,
                greeting,
                body.booking_contact or "",
            ),
        )

    _ACTIVE_BOT_IDS_CACHE.add(body.bot_id)
    logging.info("[Save Draft Bot] Admin '%s' saved draft bot '%s' (%s)", admin["username"], body.bot_id, body.business_name)
    return {
        "success": True,
        "bot_id": body.bot_id,
        "status": "draft",
    }


@router.post("/api/business-bot/deploy")
@router.post("/api/business-bot/admin/deploy")
async def api_deploy_bot(body: DeployBotRequest, admin: dict = Depends(get_current_admin)):
    """Deploys a custom business bot, strictly binding it to the current admin."""
    admin_id = admin["id"]
    summary = body.business_summary or body.summary or f"AI Customer Assistant for {body.business_name}"
    prompt = body.dynamic_prompt or body.system_prompt or ""
    greeting = body.initial_greeting or f"Hello! Welcome to {body.business_name}. I am {body.persona_name}, your personal assistant. How can I help you today?"

    with get_db_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO deployed_bots (
                bot_id, admin_id, business_name, persona_name, business_summary,
                website_url, website_data, pdf_attachments, finalized_blueprint,
                dynamic_prompt, initial_greeting, booking_contact, status, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', now())
            ON CONFLICT (bot_id) DO UPDATE SET
                business_name = EXCLUDED.business_name,
                persona_name = EXCLUDED.persona_name,
                business_summary = EXCLUDED.business_summary,
                website_url = EXCLUDED.website_url,
                website_data = EXCLUDED.website_data,
                pdf_attachments = EXCLUDED.pdf_attachments,
                finalized_blueprint = EXCLUDED.finalized_blueprint,
                dynamic_prompt = EXCLUDED.dynamic_prompt,
                initial_greeting = EXCLUDED.initial_greeting,
                booking_contact = EXCLUDED.booking_contact,
                status = 'active',
                updated_at = now();
            """,
            (
                body.bot_id,
                admin_id,
                body.business_name,
                body.persona_name,
                summary,
                body.website_url,
                json.dumps(body.website_data or {}),
                json.dumps(body.pdf_attachments or []),
                json.dumps(body.finalized_blueprint or {}),
                prompt,
                greeting,
                body.booking_contact or "",
            ),
        )

    _ACTIVE_BOT_IDS_CACHE.add(body.bot_id)
    logging.info("[Deploy Bot] Admin '%s' deployed bot '%s' (%s)", admin["username"], body.bot_id, body.business_name)
    return {
        "success": True,
        "bot_id": body.bot_id,
        "share_url": f"/bot/{body.bot_id}",
    }


class UpdateBotRequest(BaseModel):
    business_name: Optional[str] = None
    persona_name: Optional[str] = None
    business_summary: Optional[str] = None
    summary: Optional[str] = None
    website_url: Optional[str] = None
    booking_contact: Optional[str] = None
    dynamic_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    initial_greeting: Optional[str] = None
    status: Optional[str] = None


@router.put("/api/business-bot/update/{bot_id}")
@router.put("/api/business-bot/admin/bots/{bot_id}")
async def api_update_bot(bot_id: str, body: UpdateBotRequest, admin: dict = Depends(get_current_admin)):
    """STRICT OWNERSHIP GUARD: Updates only if bot belongs to this specific admin."""
    admin_id = admin["id"]
    summary_val = body.business_summary or body.summary
    prompt_val = body.dynamic_prompt or body.system_prompt

    with get_db_cursor(commit=True) as cur:
        cur.execute("SELECT admin_id FROM deployed_bots WHERE bot_id = %s LIMIT 1", (bot_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Bot not found.")

        # Strict ownership validation
        if str(row["admin_id"]) != str(admin_id):
            logging.warning("Unauthorized edit attempt: Admin %s on bot %s owned by %s", admin_id, bot_id, row["admin_id"])
            raise HTTPException(status_code=403, detail="Forbidden: You can only edit bots created by your own account.")

        cur.execute(
            """
            UPDATE deployed_bots
            SET business_name = COALESCE(%s, business_name),
                persona_name = COALESCE(%s, persona_name),
                business_summary = COALESCE(%s, business_summary),
                website_url = COALESCE(%s, website_url),
                booking_contact = COALESCE(%s, booking_contact),
                dynamic_prompt = COALESCE(%s, dynamic_prompt),
                initial_greeting = COALESCE(%s, initial_greeting),
                status = COALESCE(%s, status),
                updated_at = now()
            WHERE bot_id = %s AND admin_id = %s;
            """,
            (
                body.business_name,
                body.persona_name,
                summary_val,
                body.website_url,
                body.booking_contact,
                prompt_val,
                body.initial_greeting,
                body.status,
                bot_id,
                admin_id,
            ),
        )

    logging.info("[Bot Update] Admin '%s' updated bot '%s'", admin["username"], bot_id)
    return {"success": True, "message": "Bot updated successfully."}


@router.delete("/api/business-bot/admin/bots/{bot_id}")
@router.delete("/api/business-bot/delete/{bot_id}")
async def api_delete_bot(bot_id: str, admin: dict = Depends(get_current_admin)):
    """
    STRICT OWNERSHIP GUARD & RESILIENT TEARDOWN:
    Deletes the bot, associated Qdrant collection (biz_{bot_id}), and cascades
    to bot_chat_sessions, bot_messages, and bot_leads in PostgreSQL.
    """
    admin_id = admin["id"]
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            "SELECT bot_id, admin_id, persona_name, business_name FROM deployed_bots WHERE bot_id = %s LIMIT 1",
            (bot_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Assistant not found.")

        # Strict ownership validation
        if str(row["admin_id"]) != str(admin_id):
            logging.warning("Unauthorized delete attempt: Admin %s on bot %s owned by %s", admin_id, bot_id, row["admin_id"])
            raise HTTPException(status_code=403, detail="Forbidden: You can only delete assistants created by your own account.")

        persona_name = row["persona_name"]
        business_name = row["business_name"]

    # Step 1: Atomic PostgreSQL Deletion (foreign keys ON DELETE CASCADE automatically purge leads, chat sessions, messages)
    with get_db_cursor(commit=True) as cur:
        cur.execute(
            "DELETE FROM deployed_bots WHERE bot_id = %s AND admin_id = %s;",
            (bot_id, admin_id),
        )

    # Evict from in-memory active bot cache
    _ACTIVE_BOT_IDS_CACHE.discard(bot_id)

    # Step 2: Non-blocking background cleanup for Qdrant collection
    collection_name = f"biz_{bot_id}"
    async def _async_qdrant_cleanup(col_name: str):
        try:
            from qdrant_client import AsyncQdrantClient
            from src.constants import QDRANT_URL, QDRANT_API_KEY
            if QDRANT_URL:
                client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
                await client.delete_collection(collection_name=col_name)
                await client.close()
                logging.info("[Bot Delete] Background Qdrant collection '%s' deleted successfully", col_name)
        except Exception as qdrant_err:
            logging.warning("[Bot Delete] Background Qdrant cleanup notice for '%s': %s", col_name, qdrant_err)

    asyncio.create_task(_async_qdrant_cleanup(collection_name))

    logging.info("[Bot Delete] Admin '%s' successfully deleted bot '%s' (%s - %s)", admin["username"], bot_id, persona_name, business_name)
    return {
        "success": True,
        "message": f"Assistant '{persona_name}' deleted successfully.",
        "deleted_bot_id": bot_id,
    }


@router.get("/api/business-bot/admin/bots")
async def api_get_admin_bots(admin: dict = Depends(get_current_admin)):
    """Lists ONLY the bots created by the authenticated admin."""
    admin_id = admin["id"]
    with get_db_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT bot_id, business_name, persona_name, business_summary, dynamic_prompt,
                   website_url, booking_contact, status, created_at, updated_at
            FROM deployed_bots
            WHERE admin_id = %s
            ORDER BY updated_at DESC;
            """,
            (admin_id,),
        )
        rows = cur.fetchall() or []
        return {
            "success": True,
            "bots": [
                {
                    "bot_id": r["bot_id"],
                    "business_name": r["business_name"],
                    "persona_name": r["persona_name"],
                    "business_summary": r["business_summary"],
                    "dynamic_prompt": r.get("dynamic_prompt") or "",
                    "website_url": r.get("website_url") or "",
                    "booking_contact": r.get("booking_contact") or "",
                    "status": r.get("status") or "active",
                    "created_at": r["created_at"].isoformat() if r.get("created_at") else "",
                    "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else "",
                }
                for r in rows
            ],
        }


@router.get("/api/business-bot/admin/leads/{bot_id}")
async def api_get_bot_leads(bot_id: str, admin: dict = Depends(get_current_admin)):
    """STRICT OWNERSHIP GUARD: Returns leads for bot only if owned by this admin."""
    admin_id = admin["id"]
    with get_db_cursor(commit=False) as cur:
        cur.execute("SELECT admin_id FROM deployed_bots WHERE bot_id = %s LIMIT 1", (bot_id,))
        row = cur.fetchone()
        if not row or str(row["admin_id"]) != str(admin_id):
            raise HTTPException(status_code=403, detail="Forbidden: You do not have permission to view leads for this bot.")

        cur.execute(
            """
            SELECT id, customer_username, customer_name, customer_email, customer_phone,
                   inquiry_type, message_excerpt, created_at
            FROM bot_leads
            WHERE bot_id = %s
            ORDER BY created_at DESC;
            """,
            (bot_id,),
        )
        leads = cur.fetchall() or []
        formatted_leads = []
        for l in leads:
            email = (l.get("customer_email") or "").strip()
            phone = (l.get("customer_phone") or "").strip()
            contact_parts = []
            if email:
                contact_parts.append(email)
            if phone:
                contact_parts.append(phone)
            contact_str = ", ".join(contact_parts) if contact_parts else "Not provided"

            formatted_leads.append({
                "id": l["id"],
                "customer_username": l["customer_username"],
                "customer_name": (l.get("customer_name") or l["customer_username"]).strip(),
                "customer_email": email,
                "customer_phone": phone,
                "contact_info": contact_str,
                "inquiry_type": l["inquiry_type"],
                "inquiry_details": l.get("message_excerpt") or "Booking inquiry",
                "message_excerpt": l.get("message_excerpt"),
                "created_at": l["created_at"].isoformat() if l.get("created_at") else "",
            })
        return {
            "success": True,
            "leads": formatted_leads,
        }


# ═══════════════════════════════════════════════════════════════════
#  WEBSOCKET ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.websocket("/ws/business-bot/interview/{interview_id}")
async def ws_interview_endpoint(websocket: WebSocket, interview_id: str):
    """
    Live WebSocket connection for the Master Architect Onboarding Interview.
    Streams 8-10 turn probing sequence in raw structured JSON.
    """
    await websocket.accept()
    logging.info("[WS Interview] Accepted connection for session %s", interview_id)

    try:
        # Await initialization frame from client
        init_frame = await websocket.receive_json()
        business_name = init_frame.get("business_name", "My Business")
        persona_name = init_frame.get("persona_name", "Business Assistant")
        extracted_context = init_frame.get("extracted_context", "")

        # Create interview state machine
        interview = MasterArchitectInterview(
            business_name=business_name,
            persona_name=persona_name,
            extracted_context=extracted_context,
        )
        _active_interviews[interview_id] = interview

        # Start Turn 1
        turn_data = await interview.start()
        await websocket.send_json({
            "type": "architect_turn",
            **turn_data,
        })

        # Process turns sequentially
        while not interview.is_complete:
            data = await websocket.receive_json()
            msg_type = data.get("type", "answer")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type in ("finish", "complete"):
                result = await interview.finish()
                await websocket.send_json({
                    "type": "interview_complete",
                    "blueprint": result.get("finalized_blueprint", {}),
                    **result,
                })
                break

            user_answer = (data.get("content") or data.get("text") or data.get("answer") or "").strip()
            if not user_answer:
                continue

            result = await interview.step(user_answer)
            if result.get("interview_status") == "COMPLETE":
                await websocket.send_json({
                    "type": "interview_complete",
                    "blueprint": result.get("finalized_blueprint", {}),
                    **result,
                })
                break
            else:
                await websocket.send_json({
                    "type": "architect_turn",
                    **result,
                })

    except WebSocketDisconnect:
        logging.info("[WS Interview] Client disconnected: %s", interview_id)
    except Exception as e:
        logging.exception("[WS Interview] Unhandled error in interview %s", interview_id)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        _active_interviews.pop(interview_id, None)


@router.websocket("/ws/business-bot/chat/{bot_id}")
async def ws_customer_chat_endpoint(websocket: WebSocket, bot_id: str):
    """
    Customer WebSocket Chat with deployed bot:
    Low-latency token-by-token streaming, tool status, and lead capture.
    """
    await websocket.accept()
    logging.info("[WS Customer Chat] Accepted connection for bot '%s'", bot_id)

    try:
        # Resolve user info from cookie if present
        cookie_header = websocket.headers.get("cookie", "")
        token = None
        for item in cookie_header.split(";"):
            item = item.strip()
            if item.startswith("access_token="):
                token = item.split("=", 1)[1]
                break

        customer_username = "Customer"
        customer_id = None
        if token:
            payload = decode_access_token(token)
            if payload:
                if payload.get("username"):
                    customer_username = payload["username"]
                if payload.get("sub"):
                    try:
                        customer_id = uuid.UUID(str(payload["sub"]))
                    except Exception:
                        customer_id = None

        # Check for existing session within 72 hours (3-day inactivity window)
        active_session = None
        session_expired = False
        if customer_id or (customer_username and customer_username != "Customer"):
            try:
                with get_db_cursor(commit=False) as cur:
                    if customer_id:
                        cur.execute(
                            """
                            SELECT session_id, updated_at,
                                   (updated_at >= now() - INTERVAL '72 hours') AS is_active
                            FROM bot_chat_sessions
                            WHERE bot_id = %s AND customer_id = %s
                            ORDER BY updated_at DESC
                            LIMIT 1;
                            """,
                            (bot_id, str(customer_id)),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT session_id, updated_at,
                                   (updated_at >= now() - INTERVAL '72 hours') AS is_active
                            FROM bot_chat_sessions
                            WHERE bot_id = %s AND customer_name = %s
                            ORDER BY updated_at DESC
                            LIMIT 1;
                            """,
                            (bot_id, customer_username),
                        )
                    latest_session = cur.fetchone()
                    if latest_session:
                        if latest_session.get("is_active"):
                            active_session = latest_session
                        else:
                            session_expired = True
                            logging.info(
                                "[WS Customer Chat] Session %s for user '%s' expired (>72h inactivity)",
                                latest_session.get("session_id"),
                                customer_username,
                            )
            except Exception as e:
                logging.warning("[WS Customer Chat] Session lookup error: %s", e)

        if active_session:
            session_id = active_session["session_id"]
            chat_session = BusinessBotChatSession(
                bot_id=bot_id,
                session_id=session_id,
                customer_username=customer_username,
                customer_id=str(customer_id) if customer_id else None,
            )
            raw_history = []
            try:
                with get_db_cursor(commit=False) as cur:
                    cur.execute(
                        """
                        SELECT role, content
                        FROM bot_messages
                        WHERE session_id = %s
                        ORDER BY id ASC;
                        """,
                        (session_id,),
                    )
                    raw_history = cur.fetchall() or []
            except Exception as e:
                logging.warning("[WS Customer Chat] Could not fetch previous messages: %s", e)

            logging.info(
                "[WS Customer Chat] Resumed session %s with %d messages for user '%s'",
                session_id,
                len(raw_history),
                customer_username,
            )
            await websocket.send_json({
                "type": "resumed_history",
                "session_id": session_id,
                "resumed": True,
                "messages": [
                    {"role": r["role"], "content": r["content"]}
                    for r in raw_history
                ],
                "persona_name": chat_session.bot_data.get("persona_name"),
                "business_name": chat_session.bot_data.get("business_name"),
            })
        else:
            session_id = f"bcs_{uuid.uuid4().hex[:10]}"
            chat_session = BusinessBotChatSession(
                bot_id=bot_id,
                session_id=session_id,
                customer_username=customer_username,
                customer_id=str(customer_id) if customer_id else None,
            )
            greeting = await chat_session.get_greeting()
            logging.info(
                "[WS Customer Chat] Started fresh session %s for user '%s'",
                session_id,
                customer_username,
            )
            await websocket.send_json({
                "type": "greeting",
                "session_id": session_id,
                "resumed": False,
                "session_expired": session_expired,
                "content": greeting,
                "persona_name": chat_session.bot_data.get("persona_name"),
                "business_name": chat_session.bot_data.get("business_name"),
            })

        while True:
            frame = await websocket.receive_json()
            frame_type = frame.get("type", "user_message")

            if frame_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if frame_type in ("user_message", "query"):
                user_msg = (frame.get("content") or frame.get("query") or frame.get("text") or "").strip()
                if not user_msg:
                    continue

                async for chunk_event in chat_session.stream_reply(user_msg):
                    await websocket.send_json(chunk_event)

    except WebSocketDisconnect:
        logging.info("[WS Customer Chat] Client disconnected from bot %s", bot_id)
    except Exception as e:
        logging.exception("[WS Customer Chat] Error in bot %s", bot_id)
        try:
            await websocket.send_json({"type": "error", "message": "An error occurred during chat session."})
        except Exception:
            pass
