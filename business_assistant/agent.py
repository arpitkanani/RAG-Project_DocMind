import json
import re
from typing import AsyncGenerator, Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from business_assistant.tools.rag_tool import get_brochure_rag_tool
from src.chains.qa_chain import _build_llm
from src.database.db import get_db_cursor
from src.logger import logging

APPOINTMENT_KEYWORDS = re.compile(
    r"\b(book|appointment|schedule|consultation|visit|timing|callback|call me|sign up|buy|purchase|order)\b",
    re.IGNORECASE,
)


def _check_and_log_lead(bot_id: str, customer_username: str, user_message: str) -> Optional[dict]:
    """Detects purchase/appointment intent and records a lead entry in PostgreSQL."""
    if APPOINTMENT_KEYWORDS.search(user_message):
        try:
            inquiry_type = "appointment" if re.search(r"\b(book|appointment|schedule|consultation|visit)\b", user_message, re.IGNORECASE) else "purchase_intent"
            with get_db_cursor(commit=True) as cur:
                cur.execute(
                    """
                    INSERT INTO bot_leads (bot_id, customer_username, inquiry_type, message_excerpt)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, created_at;
                    """,
                    (bot_id, customer_username or "Customer", inquiry_type, user_message.strip()[:200]),
                )
                row = cur.fetchone()
                logging.info(
                    "[Lead Capture] Logged lead #%s for bot '%s' from user '%s' (%s)",
                    row.get("id") if row else "N/A",
                    bot_id,
                    customer_username,
                    inquiry_type,
                )
                return {
                    "captured": True,
                    "inquiry_type": inquiry_type,
                }
        except Exception as e:
            logging.warning("[Lead Capture] Could not record lead: %s", e)
    return None


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
        lead_info = _check_and_log_lead(self.bot_id, self.customer_username, user_message)
        if lead_info:
            yield {
                "type": "lead_alert",
                "lead_captured": True,
                "inquiry_type": lead_info["inquiry_type"],
                "customer_name": self.customer_username,
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
            "5. CLOSING RULE: Only provide a closing or goodbye message if the customer explicitly says goodbye or thank you."
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

        # Save AI reply to PostgreSQL
        if full_reply.strip():
            try:
                with get_db_cursor(commit=True) as cur:
                    cur.execute(
                        """
                        INSERT INTO bot_messages (session_id, role, content)
                        VALUES (%s, 'ai', %s);
                        """,
                        (self.session_id, full_reply),
                    )
            except Exception as e:
                logging.warning("[Business Agent] Failed saving AI message: %s", e)

        yield {"type": "done", "full_reply": full_reply}
