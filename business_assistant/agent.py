import json
import re
from typing import AsyncGenerator, Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from business_assistant.tools.rag_tool import get_brochure_rag_tool
from src.chains.qa_chain import _build_llm
from src.database.db import get_db_cursor
from src.logger import logging

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,9}")
NAME_REGEX = re.compile(r"(?:my name is|i am|i'm|this is|call me|name:)\s+([A-Za-z\s]{2,30})", re.IGNORECASE)

APPOINTMENT_KEYWORDS = re.compile(
    r"\b(book|appointment|schedule|consultation|visit|timing|callback|call me|sign up|buy|purchase|order|quote|price|pricing)\b",
    re.IGNORECASE,
)


def _extract_and_log_lead(
    bot_id: str,
    session_id: str,
    customer_username: str,
    user_message: str,
) -> Optional[dict]:
    """
    Detects customer contact details (Name, Email, Phone) or appointment/purchase intent
    and records or updates a lead entry in PostgreSQL (bot_leads), correlated by session_id.
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
    if not extracted_name and customer_username and customer_username.lower() not in ("customer", "guest"):
        extracted_name = customer_username

    has_intent = bool(APPOINTMENT_KEYWORDS.search(user_message))

    # Only log a lead if user provided contact details OR expressed explicit booking/purchase intent
    if not (extracted_email or extracted_phone or has_intent):
        return None

    # Determine inquiry type
    if re.search(r"\b(book|appointment|schedule|consultation|visit)\b", user_message, re.IGNORECASE):
        inquiry_type = "appointment"
    elif re.search(r"\b(buy|purchase|order)\b", user_message, re.IGNORECASE):
        inquiry_type = "purchase_intent"
    elif re.search(r"\b(quote|price|pricing|cost)\b", user_message, re.IGNORECASE):
        inquiry_type = "pricing_quote"
    elif extracted_email or extracted_phone:
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

            if existing:
                lead_id = existing["id"]
                merged_name = extracted_name or existing.get("customer_name") or customer_username or "Customer"
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
                final_name = extracted_name or customer_username or "Customer"
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
                        customer_username or "Customer",
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
                ", ".join(contact_parts) or "Pending",
                inquiry_type,
            )
            return {
                "captured": True,
                "inquiry_type": inquiry_type,
                "customer_name": merged_name,
                "contact_info": ", ".join(contact_parts) if contact_parts else None,
                "has_email": bool(merged_email),
                "has_phone": bool(merged_phone),
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

    async def _get_history_and_summary(self) -> tuple[str, List[Any]]:
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
                summary_resp = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
                new_summary = summary_resp.content if hasattr(summary_resp, "content") else str(summary_resp)
                new_summary = new_summary.strip()

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

    async def stream_reply(self, user_message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes conversation turn:
        1. Checks lead capture intent.
        2. Retrieves brochure context directly and history summary.
        3. Enforces anti-loop and anti-greeting constraints.
        4. Streams tokens from LLM fallback hierarchy.
        5. Saves messages to PostgreSQL.
        """
        lead_info = _extract_and_log_lead(self.bot_id, self.session_id, self.customer_username, user_message)
        if lead_info:
            yield {
                "type": "lead_alert",
                "lead_captured": True,
                "inquiry_type": lead_info.get("inquiry_type"),
                "customer_name": lead_info.get("customer_name") or self.customer_username,
                "contact_info": lead_info.get("contact_info"),
            }

        # Save user message to PostgreSQL
        try:
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
        except Exception as e:
            logging.warning("[Business Agent] Failed saving human message: %s", e)

        # Check if brochure lookup is available
        tool_context = ""
        try:
            tool_context = await self.rag_tool.ainvoke(user_message)
            if tool_context and "No brochures available" not in tool_context and "Collection not found" not in tool_context:
                yield {
                    "type": "tool_status",
                    "status": "Consulting business brochures...",
                    "message": "Consulting business brochures...",
                }
                logging.info("[Business Agent] Retrieved %d chars of brochure context", len(tool_context))
        except Exception as e:
            logging.warning("[Business Agent] Tool invocation note: %s", e)

        # Retrieve conversation history (sliding-window 10 recent + older summary)
        prior_summary, history_messages = await self._get_history_and_summary()

        # Build prompt messages
        system_content = self.bot_data.get("dynamic_prompt") or f"You are {self.bot_data.get('persona_name')} representing {self.bot_data.get('business_name')}."
        if tool_context and "No brochures available" not in tool_context and "Collection not found" not in tool_context:
            system_content += f"\n\n### VERIFIED BROCHURE KNOWLEDGE:\n{tool_context}"

        if prior_summary:
            system_content += f"\n\n### PRIOR CONVERSATION SUMMARY (Earlier Turns):\n{prior_summary}"

        system_content += (
            "\n\n### CRITICAL CONVERSATION FLOW RULES:\n"
            "1. NO REPEATED GREETINGS: The initial greeting was already delivered on turn 1. NEVER start responses with 'Hello', 'Hi', 'Welcome to...', or re-introduce yourself. Jump straight into answering the user's specific query.\n"
            "2. ANTI-LOOP RULE: Never repeat the same question or menu of options you asked in previous turns. If the customer mentions an interest or selects a product, immediately provide specific facts, instructions, benefits, and pricing from the verified knowledge.\n"
            "3. DIRECT ANSWERS & CLEAN STRUCTURE: Answer questions clearly and definitively with well-structured Markdown. Use clean headings (###), bold key terms, and bullet points (-). NEVER output raw HTML tags like <br>, <br/>, or <p> - always use clean Markdown formatting with double newlines between paragraphs.\n"
            "4. NO TOOL CALLS: Answer directly using the verified knowledge provided. Do NOT output any XML or JSON tool call tags.\n"
            "5. CLOSING RULE: Only provide a closing or goodbye message if the customer explicitly says goodbye or thank you.\n"
            "6. PRIVACY SHIELD FOR CONTACT INFO: If the customer provides their phone number, email address, or contact details, acknowledge receipt politely (e.g. 'Thank you [Customer Name], our team has noted your contact details and will follow up shortly.'). STRICT MANDATORY RULE: NEVER repeat, echo, quote, or display their raw phone number or email address back in your response.\n"
            "7. OUT-OF-KNOWLEDGE BOUNDARY: If the customer asks questions outside your verified brochure knowledge and business offerings, do NOT invent or guess details. Follow your configured out-of-scope policy: explain politely that this is outside your verified scope, and invite them to leave their contact details or consult official business channels."
        )

        messages = [
            SystemMessage(content=system_content),
            *history_messages,
            HumanMessage(content=user_message),
        ]

        full_reply = ""
        try:
            # Stream tokens from LLM
            async for chunk in self.llm.astream(messages):
                text_chunk = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text_chunk:
                    full_reply += text_chunk
                    yield {"type": "token", "token": text_chunk, "chunk": text_chunk}
        except Exception as e:
            logging.exception("[Business Agent] Streaming generation error")
            yield {"type": "error", "message": "The assistant is temporarily unavailable. Please try again in a few moments."}
            return

        # Save AI reply to PostgreSQL (with defensive privacy redaction against echoing contact info)
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

        yield {"type": "done", "full_reply": full_reply}
