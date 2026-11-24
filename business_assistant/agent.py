import asyncio
import json
import re
import time
from typing import AsyncGenerator, Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from business_assistant.tools.rag_tool import get_brochure_rag_tool
from src.chains.qa_chain import _build_llm
from src.database.db import get_db_cursor
from src.logger import logging, log_latency
from src.utils.perf import log_perf_kv, mask_pii


EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,9}")
NAME_REGEX = re.compile(r"(?:my name is|i am|i'm|this is|call me|name:)\s+([A-Za-z\s]{2,30})", re.IGNORECASE)

# Dynamic booking & appointment intent across all industries (Clinics, Salons, Restaurants, Auto, B2B, etc.)
BOOKING_INTENT_REGEX = re.compile(
    r"\b("
    r"book\s+(?:an?\s+)?(?:appointment|session|slot|table|visit|test\s*drive|consultation|service|call|demo)|"
    r"schedule\s+(?:an?\s+)?(?:appointment|call|meeting|visit|consultation|test\s*drive|session|demo)|"
    r"reserve\s+(?:a\s+)?(?:table|seat|slot|room|ticket|vehicle)|"
    r"make\s+an?\s+appointment|"
    r"fix\s+an?\s+appointment|"
    r"set\s+up\s+an?\s+appointment|"
    r"need\s+an?\s+appointment|"
    r"appointment\s+booking"
    r")\b",
    re.IGNORECASE,
)

# Callback & direct contact request
CALLBACK_INTENT_REGEX = re.compile(
    r"\b("
    r"call\s+me(?:\s+back)?|"
    r"callback|"
    r"contact\s+me|"
    r"have\s+(?:someone|an?\s+agent|sales|your\s+team)\s+call\s+me|"
    r"reach\s+out(?:\s+to\s+me)?|"
    r"get\s+in\s+touch(?:\s+with\s+me)?"
    r")\b",
    re.IGNORECASE,
)

# Purchase & ordering intent
PURCHASE_INTENT_REGEX = re.compile(
    r"\b("
    r"want\s+to\s+(?:buy|purchase|order|reserve|hire|subscribe)|"
    r"ready\s+to\s+(?:buy|purchase|order|book)|"
    r"place\s+an?\s+order|"
    r"how\s+can\s+i\s+(?:buy|order|book|reserve)|"
    r"sign\s+me\s+up|"
    r"order\s+now"
    r")\b",
    re.IGNORECASE,
)

# Formal quote request
QUOTE_REQUEST_REGEX = re.compile(
    r"\b("
    r"(?:send|give|email|share)\s+(?:me\s+)?(?:a\s+)?(?:formal\s+)?(?:quotation|quote|estimate|pricing\s+proposal)|"
    r"request\s+(?:a\s+)?(?:quote|quotation|estimate)|"
    r"on-road\s+price\s+quote|"
    r"get\s+(?:a\s+)?quote"
    r")\b",
    re.IGNORECASE,
)

ADMIN_TEST_USERNAMES = {"admin", "admin1", "admin2", "admin3", "customer", "guest", "tester", "test", "anonymous", "user"}


def _extract_and_log_lead(
    bot_id: str,
    session_id: str,
    customer_username: str,
    user_message: str,
) -> Optional[dict]:
    """
    Detects customer contact details (Name, Email, Phone) or explicit purchase/booking intent
    dynamically across all business verticals, and records or updates a lead entry in PostgreSQL (bot_leads).
    General informational queries (e.g. asking pricing, specifications, hours) are NOT treated as leads.
    """
    # 1. Extract Email
    email_match = EMAIL_REGEX.search(user_message)
    extracted_email = email_match.group(0).strip() if email_match else None

    # 2. Extract Phone (must contain at least 7 digits to avoid matching small numbers or years)
    extracted_phone = None
    phone_candidates = PHONE_REGEX.findall(user_message)
    for cand in phone_candidates:
        cand_digits = re.sub(r"\D", "", cand)
        if 7 <= len(cand_digits) <= 15:
            extracted_phone = cand.strip()
            break

    # 3. Extract Name
    extracted_name = None
    name_match = NAME_REGEX.search(user_message)
    if name_match:
        cand_name = name_match.group(1).strip()
        cand_name = re.sub(r"\b(and|or|my|email|phone|here|is)\b.*$", "", cand_name, flags=re.IGNORECASE).strip()
        if 2 <= len(cand_name) <= 40:
            extracted_name = cand_name

    # Only assign customer_username if it is a genuine username, not an admin/tester handle
    clean_username = (customer_username or "").strip()
    is_admin_or_test = clean_username.lower() in ADMIN_TEST_USERNAMES or clean_username.lower().startswith("admin")
    if not extracted_name and clean_username and not is_admin_or_test:
        extracted_name = clean_username

    # 4. Check Explicit High Intent (Dynamic across all business verticals)
    is_booking = bool(BOOKING_INTENT_REGEX.search(user_message))
    is_callback = bool(CALLBACK_INTENT_REGEX.search(user_message))
    is_purchase = bool(PURCHASE_INTENT_REGEX.search(user_message))
    is_quote = bool(QUOTE_REQUEST_REGEX.search(user_message))

    has_contact = bool(extracted_email or extracted_phone)
    has_high_intent = is_booking or is_callback or is_purchase or is_quote

    # Option 2 followed by Option 1:
    # Only capture lead if user provided contact details OR expressed explicit booking/purchase/quote intent
    if not (has_contact or has_high_intent):
        return None

    # Determine inquiry type accurately
    if is_booking:
        inquiry_type = "appointment"
    elif is_callback:
        inquiry_type = "callback_request"
    elif is_purchase:
        inquiry_type = "purchase_intent"
    elif is_quote:
        inquiry_type = "pricing_quote"
    elif has_contact:
        inquiry_type = "contact_lead"
    else:
        inquiry_type = "general_inquiry"

    try:
        with get_db_cursor(commit=True) as cur:
            # Check for existing lead record for this session
            cur.execute(
                """
                SELECT id, customer_name, customer_email, customer_phone, message_excerpt 
                FROM bot_leads 
                WHERE bot_id = %s AND session_id = %s 
                ORDER BY created_at DESC LIMIT 1;
                """,
                (bot_id, session_id),
            )
            existing = cur.fetchone()

            clean_excerpt = user_message.strip()[:250]
            # Redact email and phone from stored excerpt for database privacy
            if extracted_email:
                clean_excerpt = clean_excerpt.replace(extracted_email, "[email provided]")
            if extracted_phone:
                clean_excerpt = clean_excerpt.replace(extracted_phone, "[phone provided]")

            fallback_name = "Prospective Customer"
            if existing:
                lead_id = existing["id"]
                merged_name = extracted_name or existing.get("customer_name") or fallback_name
                merged_email = extracted_email or existing.get("customer_email")
                merged_phone = extracted_phone or existing.get("customer_phone")
                cur.execute(
                    """
                    UPDATE bot_leads 
                    SET customer_name = %s, customer_email = %s, customer_phone = %s,
                        inquiry_type = %s, message_excerpt = %s
                    WHERE id = %s;
                    """,
                    (merged_name, merged_email, merged_phone, inquiry_type, clean_excerpt, lead_id),
                )
            else:
                final_name = extracted_name or fallback_name
                merged_name = final_name
                merged_email = extracted_email
                merged_phone = extracted_phone
                cur.execute(
                    """
                    INSERT INTO bot_leads (
                        bot_id, customer_username, customer_name, customer_email, 
                        customer_phone, session_id, inquiry_type, message_excerpt
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        bot_id,
                        clean_username or "Customer",
                        final_name,
                        extracted_email,
                        extracted_phone,
                        session_id,
                        inquiry_type,
                        clean_excerpt,
                    ),
                )
                row = cur.fetchone()
                lead_id = row.get("id") if row else "new"

            contact_parts = []
            if merged_email:
                contact_parts.append(merged_email)
            if merged_phone:
                contact_parts.append(merged_phone)

            logging.info(
                "[Lead Capture] Recorded/Updated lead #%s for bot '%s' (Name: %s | Contact: %s | Type: %s)",
                lead_id,
                bot_id,
                merged_name,
                ", ".join(contact_parts) or "Pending (Awaiting user details)",
                inquiry_type,
            )
            return {
                "captured": True,
                "inquiry_type": inquiry_type,
                "customer_name": merged_name,
                "contact_info": ", ".join(contact_parts) if contact_parts else None,
                "has_email": bool(merged_email),
                "has_phone": bool(merged_phone),
                "has_contact": bool(merged_email or merged_phone),
                "is_high_intent": has_high_intent,
                "email": merged_email,
                "phone": merged_phone,
            }
    except Exception as e:
        logging.warning("[Lead Capture] Could not record lead: %s", e)
        return None


def _check_and_log_lead(bot_id: str, customer_username: str, user_message: str, session_id: str = "") -> Optional[dict]:
    """Backward compatibility wrapper for _extract_and_log_lead."""
    return _extract_and_log_lead(bot_id=bot_id, session_id=session_id, customer_username=customer_username, user_message=user_message)


class BusinessBotChatSession:
    """Manages chat execution for a deployed bot instance with hierarchical memory and token streaming."""

    def __init__(self, bot_id: str, session_id: str, customer_username: str = "Customer", customer_id: Optional[str] = None):
        self.bot_id = bot_id
        self.session_id = session_id
        self.customer_username = customer_username
        self.customer_id = customer_id
        self.bot_data = self._load_bot()
        self._rag_tool = None
        self._llm = None

    @property
    def rag_tool(self):
        if self._rag_tool is None:
            self._rag_tool = get_brochure_rag_tool(self.bot_id)
        return self._rag_tool

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _build_llm()
        return self._llm

    def _load_bot(self) -> Dict[str, Any]:
        """Loads deployed bot metadata and dynamic system prompt from PostgreSQL."""
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT bot_id, business_name, persona_name, dynamic_prompt,
                       initial_greeting, booking_contact, status
                FROM deployed_bots
                WHERE bot_id = %s
                LIMIT 1;
                """,
                (self.bot_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Deployed bot '{self.bot_id}' not found.")
            return dict(row)

    async def get_greeting(self) -> str:
        """Returns the bot's initial greeting."""
        greeting = self.bot_data.get("initial_greeting")
        if not greeting:
            greeting = f"Hello! Welcome to {self.bot_data.get('business_name')}. I am {self.bot_data.get('persona_name')}, your personal assistant. How can I help you today?"
        return greeting

    async def _get_history_and_summary(self, run_config: Optional[RunnableConfig] = None) -> tuple[str, List[Any]]:
        """
        Retrieves conversation history using the sliding-window + progressive summarization pattern:
        - If <= 10 messages: all are passed verbatim as history_messages.
        - If > 10 messages:
          * Older messages (0 to len-10) are progressively summarized and stored in bot_chat_sessions.
          * Recent 10 messages are passed verbatim as history_messages.
        Returns (prior_summary_str, history_messages).
        """
        raw_messages = []
        existing_summary = ""
        try:
            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    "SELECT conversation_summary FROM bot_chat_sessions WHERE session_id = %s LIMIT 1;",
                    (self.session_id,),
                )
                sess_row = cur.fetchone()
                if sess_row and sess_row.get("conversation_summary"):
                    existing_summary = sess_row["conversation_summary"].strip()

                cur.execute(
                    """
                    SELECT role, content
                    FROM bot_messages
                    WHERE session_id = %s
                    ORDER BY id ASC;
                    """,
                    (self.session_id,),
                )
                raw_messages = cur.fetchall() or []
        except Exception as e:
            logging.warning("[Business Agent] Could not load message history: %s", e)
            return existing_summary, []

        total_msgs = len(raw_messages)
        if total_msgs <= 10:
            history_objs = []
            for r in raw_messages:
                if r["role"] == "human":
                    history_objs.append(HumanMessage(content=r["content"]))
                elif r["role"] == "ai":
                    history_objs.append(AIMessage(content=r["content"]))
            return existing_summary, history_objs

        # Total messages > 10
        older_messages = raw_messages[:-10]
        recent_messages = raw_messages[-10:]

        older_text = "\n".join(
            f"{'Customer' if m['role'] == 'human' else self.bot_data.get('persona_name', 'Assistant')}: {m['content']}"
            for m in older_messages
        )

        new_summary = existing_summary
        if older_text:
            try:
                summary_prompt = (
                    f"Summarize the following earlier chat history between a customer and the assistant. "
                    f"Focus on key products mentioned, user questions answered, pricing discussed, and customer preferences. "
                    f"Keep it concise (1-2 short paragraphs or bullet points). Do not invent details.\n\n"
                    f"PREVIOUS SUMMARY:\n{existing_summary or 'None'}\n\n"
                    f"NEW MESSAGES TO INCORPORATE:\n{older_text[-3000:]}"
                )
                t_sum_start = time.perf_counter()
                summary_resp = await self.llm.ainvoke([HumanMessage(content=summary_prompt)], config=run_config)
                sum_duration_ms = (time.perf_counter() - t_sum_start) * 1000.0
                new_summary = summary_resp.content if hasattr(summary_resp, "content") else str(summary_resp)
                new_summary = new_summary.strip()
                log_latency(
                    "CUSTOM_BOT_SUMMARY",
                    f"Session: '{self.session_id}' | Summarized {len(older_messages)} older turns in {sum_duration_ms:.1f}ms",
                )

                with get_db_cursor(commit=True) as cur:
                    cur.execute(
                        "UPDATE bot_chat_sessions SET conversation_summary = %s WHERE session_id = %s;",
                        (new_summary, self.session_id),
                    )
            except Exception as sum_err:
                logging.warning("[Business Agent] Summarization notice: %s", sum_err)

        history_objs = []
        for r in recent_messages:
            if r["role"] == "human":
                history_objs.append(HumanMessage(content=r["content"]))
            elif r["role"] == "ai":
                history_objs.append(AIMessage(content=r["content"]))

        return new_summary, history_objs

    @traceable(name="BusinessBot.stream_reply", run_type="chain")
    async def stream_reply(self, user_message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes conversation turn:
        1. Checks lead capture intent.
        2. Retrieves brochure context directly and history summary.
        3. Enforces anti-loop and anti-greeting constraints.
        4. Streams tokens from LLM fallback hierarchy.
        5. Saves messages to PostgreSQL.
        6. Logs fine-grained response latency across all steps to logs/latency.log.
        """
        t_turn_start = time.perf_counter()

        # Build standardized LangSmith RunnableConfig for custom bot turn
        run_config: RunnableConfig = {
            "configurable": {
                "thread_id": self.session_id,
                "session_id": self.session_id,
                "bot_id": self.bot_id,
                "customer_id": self.customer_id or "anonymous",
            },
            "metadata": {
                "bot_id": self.bot_id,
                "business_name": self.bot_data.get("business_name", ""),
                "persona_name": self.bot_data.get("persona_name", ""),
                "session_id": self.session_id,
                "customer_username": self.customer_username,
                "user_query": mask_pii(user_message[:120]),
                "channel": "customer_chat_websocket",
            },
            "run_name": f"Custom Bot [{self.bot_data.get('business_name', 'Biz')}]: {mask_pii(user_message[:35])}",
            "tags": [
                "custom_bot",
                "business_assistant",
                f"bot:{self.bot_id}",
                f"session:{self.session_id}",
                f"business:{self.bot_data.get('business_name', 'Biz')}",
            ],
        }

        t_first_event: Optional[float] = None

        # Step 1: Lead capture
        t_lead_start = time.perf_counter()
        lead_info = _extract_and_log_lead(self.bot_id, self.session_id, self.customer_username, user_message)
        lead_ms = (time.perf_counter() - t_lead_start) * 1000.0

        if lead_info:
            if t_first_event is None:
                t_first_event = time.perf_counter()
            yield {
                "type": "lead_alert",
                "lead_captured": True,
                "inquiry_type": lead_info.get("inquiry_type"),
                "customer_name": lead_info.get("customer_name") or "Prospective Customer",
                "contact_info": lead_info.get("contact_info"),
                "has_contact": lead_info.get("has_contact", False),
                "is_high_intent": lead_info.get("is_high_intent", False),
            }

        # Fix 6: Immediate perceived speed event (< 20ms)
        if t_first_event is None:
            t_first_event = time.perf_counter()
        yield {"type": "tool_status", "status": "searching", "message": "Searching brochures..."}

        # Step 2, 3, 4: Concurrent Execution of DB User Insert, Brochure RAG, and History Retrieval (Fix 3)
        t_concurrent_start = time.perf_counter()

        async def _fetch_rag():
            try:
                return await self.rag_tool.ainvoke(user_message, config=run_config)
            except Exception as e:
                logging.warning("[Business Agent] Tool invocation note: %s", e)
                return ""

        async def _fetch_history():
            try:
                return await self._get_history_and_summary(run_config=run_config)
            except Exception as e:
                logging.warning("[Business Agent] History retrieval note: %s", e)
                return "", []

        async def _save_user_db():
            t0 = time.perf_counter()
            try:
                def _sync_insert():
                    with get_db_cursor(commit=True) as cur:
                        cur.execute(
                            """
                            INSERT INTO bot_chat_sessions (session_id, bot_id, customer_id, customer_name, turn_count, updated_at)
                            VALUES (%s, %s, %s, %s, 1, now())
                            ON CONFLICT (session_id) DO UPDATE
                            SET turn_count = bot_chat_sessions.turn_count + 1, updated_at = now();
                            """,
                            (self.session_id, self.bot_id, self.customer_id, self.customer_username),
                        )
                        cur.execute(
                            """
                            INSERT INTO bot_messages (session_id, role, content)
                            VALUES (%s, 'human', %s);
                            """,
                            (self.session_id, user_message),
                        )
                await asyncio.to_thread(_sync_insert)
            except Exception as e:
                logging.warning("[Business Agent] Failed saving human message: %s", e)
            return (time.perf_counter() - t0) * 1000.0

        rag_task = asyncio.create_task(_fetch_rag())
        hist_task = asyncio.create_task(_fetch_history())
        db_u_task = asyncio.create_task(_save_user_db())

        tool_context, (prior_summary, history_messages), db_user_ms = await asyncio.gather(
            rag_task, hist_task, db_u_task
        )
        t_rag_done = time.perf_counter()
        rag_ms = (t_rag_done - t_concurrent_start) * 1000.0
        history_ms = 0.0

        if tool_context and "No brochures available" not in tool_context and "Collection not found" not in tool_context:
            logging.info("[Business Agent] Retrieved %d chars of brochure context in %.1fms", len(tool_context), rag_ms)

        # Prevent duplicate user turn if DB insert completed before history query
        if history_messages and isinstance(history_messages[-1], HumanMessage) and history_messages[-1].content.strip() == user_message.strip():
            history_messages.pop()

        # Fix 1 & Fix 5: Token Capping & Prompt Budget Protection
        # Prevents provider TPM rate-limit stalls (429s) and accelerates LLM TTFT (< 1.5s)
        if tool_context and len(tool_context) > 2000:
            tool_context = tool_context[:2000].rstrip() + "\n\n[Brochure context capped for brevity]"
        if len(history_messages) > 4:
            history_messages = history_messages[-4:]

        # Dynamic prompt budget protection: if stored prompt is oversized (> 3200 chars),
        # compress to preserve persona identity, voice, and guardrails without blowing up token limit
        base_prompt = self.bot_data.get("dynamic_prompt") or f"You are {self.bot_data.get('persona_name')} representing {self.bot_data.get('business_name')}."
        if len(base_prompt) > 3200:
            base_prompt = base_prompt[:2200].rstrip() + "\n\n[... Note: Core product specs are retrieved dynamically from verified brochure context ...]\n\n" + base_prompt[-800:].lstrip()

        # Build prompt messages
        system_content = base_prompt
        if tool_context and "No brochures available" not in tool_context and "Collection not found" not in tool_context:
            system_content += f"\n\n### VERIFIED BROCHURE KNOWLEDGE:\n{tool_context}"

        if prior_summary:
            system_content += f"\n\n### PRIOR CONVERSATION SUMMARY (Earlier Turns):\n{prior_summary[:400]}"

        if "CRITICAL CONVERSATION FLOW RULES" not in system_content and "ANTI-LOOP" not in system_content:
            system_content += (
                "\n\n### CONVERSATION RULES:\n"
                "1. NO REPEATED GREETINGS: The initial greeting was already delivered on turn 1. NEVER start responses with 'Hello', 'Hi', or introductions. Jump straight into answering.\n"
                "2. DIRECT ANSWERS & CLEAN STRUCTURE: Answer questions clearly with structured Markdown (headings, bullet points). NEVER output raw HTML tags.\n"
                "3. PRIVACY SHIELD FOR CONTACT INFO: If the customer provides phone/email, acknowledge receipt politely, but NEVER repeat, echo, or quote their raw contact details back.\n"
                "4. OUT-OF-KNOWLEDGE BOUNDARY: If queried outside verified brochure knowledge, politely decline and offer to take contact details for team follow-up."
            )

        # Prompt size calculations
        sys_chars = len(system_content)
        sys_tokens = sys_chars // 4
        rag_chars = len(tool_context) if tool_context else 0
        rag_tokens = rag_chars // 4
        hist_chars = sum(len(m.content) for m in history_messages) + len(prior_summary or "")
        hist_tokens = hist_chars // 4
        user_chars = len(user_message)
        user_tokens = user_chars // 4
        total_input_chars = sys_chars + hist_chars + user_chars
        total_input_tokens = total_input_chars // 4

        messages = [
            SystemMessage(content=system_content),
            *history_messages,
            HumanMessage(content=user_message),
        ]

        # Step 5: Stream tokens from LLM and measure TTFT
        full_reply = ""
        first_token_time = None
        turn_ttft_ms = 0.0
        llm_ttft_ms = 0.0
        token_count = 0
        t_llm_start = time.perf_counter()

        try:
            async for chunk in self.llm.astream(messages, config=run_config):
                text_chunk = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text_chunk:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        llm_ttft_ms = (first_token_time - t_llm_start) * 1000.0
                        turn_ttft_ms = (first_token_time - t_turn_start) * 1000.0
                        if t_first_event is None:
                            t_first_event = first_token_time
                    token_count += 1
                    full_reply += text_chunk
                    yield {"type": "token", "token": text_chunk, "chunk": text_chunk}
        except Exception as e:
            logging.exception("[Business Agent] Streaming generation error")
            yield {"type": "error", "message": "The assistant is temporarily unavailable. Please try again in a few moments."}
            return

        llm_stream_ms = (time.perf_counter() - t_llm_start) * 1000.0
        if first_token_time is None:
            llm_ttft_ms = llm_stream_ms
            turn_ttft_ms = (time.perf_counter() - t_turn_start) * 1000.0
        stream_ms = (time.perf_counter() - first_token_time) * 1000.0 if first_token_time else 0.0

        # Step 6: Save AI reply to PostgreSQL (with defensive privacy redaction)
        t_db_ai_start = time.perf_counter()
        if full_reply.strip():
            saved_reply = full_reply
            if lead_info:
                if lead_info.get("email") and lead_info["email"] in saved_reply:
                    saved_reply = saved_reply.replace(lead_info["email"], "[contact info recorded]")
                if lead_info.get("phone") and lead_info["phone"] in saved_reply:
                    saved_reply = saved_reply.replace(lead_info["phone"], "[contact info recorded]")
            try:
                with get_db_cursor(commit=True) as cur:
                    cur.execute(
                        """
                        INSERT INTO bot_messages (session_id, role, content)
                        VALUES (%s, 'ai', %s);
                        """,
                        (self.session_id, saved_reply),
                    )
            except Exception as e:
                logging.warning("[Business Agent] Failed saving AI message: %s", e)
        db_ai_ms = (time.perf_counter() - t_db_ai_start) * 1000.0

        # Step 7: Calculate final metrics, log to latency.log, and yield done event
        t_turn_end = time.perf_counter()
        total_duration_ms = (t_turn_end - t_turn_start) * 1000.0
        total_db_ms = db_user_ms + db_ai_ms
        tokens_per_sec = (token_count / (stream_ms / 1000.0)) if stream_ms > 0 else 0.0

        first_event_ms = (t_first_event - t_turn_start) * 1000.0 if t_first_event else 0.0
        rag_done_rel_ms = (t_rag_done - t_turn_start) * 1000.0
        llm_sent_rel_ms = (t_llm_start - t_turn_start) * 1000.0
        first_token_rel_ms = (first_token_time - t_turn_start) * 1000.0 if first_token_time else 0.0

        # Phase 1 Key=Value PERF telemetry
        log_perf_kv(
            "chat_turn",
            bot_id=self.bot_id,
            turn_ttft_ms=round(turn_ttft_ms, 1),
            llm_ttft_ms=round(llm_ttft_ms, 1),
            rag_ms=round(rag_ms, 1),
            llm_ms=round(llm_stream_ms, 1),
            stream_ms=round(stream_ms, 1),
            history_ms=round(history_ms, 1),
            db_user_ms=round(db_user_ms, 1),
            db_ai_ms=round(db_ai_ms, 1),
            total_ms=round(total_duration_ms, 1),
            tokens=token_count,
            tok_sec=round(tokens_per_sec, 1),
        )
        log_perf_kv(
            "prompt_breakdown",
            sys_chars=sys_chars,
            sys_tokens=sys_tokens,
            rag_chars=rag_chars,
            rag_tokens=rag_tokens,
            hist_chars=hist_chars,
            hist_tokens=hist_tokens,
            user_chars=user_chars,
            user_tokens=user_tokens,
            total_input_chars=total_input_chars,
            total_input_tokens=total_input_tokens,
        )
        log_perf_kv(
            "timeline",
            t_request=0.0,
            t_first_event_ms=round(first_event_ms, 1),
            t_rag_done_ms=round(rag_done_rel_ms, 1),
            t_llm_sent_ms=round(llm_sent_rel_ms, 1),
            t_first_token_ms=round(first_token_rel_ms, 1),
            t_done_ms=round(total_duration_ms, 1),
        )
        log_perf_kv(
            "db",
            lead_ms=round(lead_ms, 1),
            user_insert_ms=round(db_user_ms, 1),
            hist_load_ms=round(history_ms, 1),
            ai_insert_ms=round(db_ai_ms, 1),
            cached=0,
            concurrent_with_rag=1,
        )

        latency_stats = {
            "total_ms": round(total_duration_ms, 1),
            "ttft_ms": round(turn_ttft_ms, 1),
            "turn_ttft_ms": round(turn_ttft_ms, 1),
            "llm_ttft_ms": round(llm_ttft_ms, 1),
            "rag_ms": round(rag_ms, 1),
            "llm_ms": round(llm_stream_ms, 1),
            "stream_ms": round(stream_ms, 1),
            "history_ms": round(history_ms, 1),
            "db_ms": round(total_db_ms, 1),
            "tokens": token_count,
            "tokens_per_sec": round(tokens_per_sec, 1),
        }

        # Structured latency log entry directly into logs/latency.log and dated session log
        log_latency(
            "CUSTOM_BOT",
            f"Bot: '{self.bot_id}' | Business: '{self.bot_data.get('business_name', '')}' | Session: '{self.session_id}' | "
            f"Total: {total_duration_ms:.1f}ms | TTFT: {turn_ttft_ms:.1f}ms (LLM TTFT: {llm_ttft_ms:.1f}ms) | RAG: {rag_ms:.1f}ms | "
            f"LLM: {llm_stream_ms:.1f}ms ({token_count} chunks @ {tokens_per_sec:.1f} tok/s) | "
            f"History: {history_ms:.1f}ms | DB: {total_db_ms:.1f}ms"
        )

        yield {
            "type": "done",
            "full_reply": full_reply,
            "latency": latency_stats,
        }

