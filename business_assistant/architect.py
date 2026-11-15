import json
import re
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from business_assistant.prompts import MASTER_ARCHITECT_SYSTEM_PROMPT, get_azure_or_fallback_llm
from src.logger import logging


class MasterArchitectInterview:
    """
    State machine that runs the live WebSocket interview with the business owner,
    probing for the 8-10 critical business vectors and outputting structured JSON.
    """

    def __init__(
        self,
        business_name: str,
        persona_name: str,
        extracted_context: str,
    ):
        self.business_name = business_name
        self.persona_name = persona_name
        self.extracted_context = (extracted_context or "").strip() or "No website or brochure context provided."
        self.llm = get_azure_or_fallback_llm()
        self.turn_count = 0
        self.messages: List[Any] = []
        self.finalized_blueprint: Dict[str, Any] = {}
        self.is_complete = False

        # Build initial system prompt with injected context
        formatted_system = MASTER_ARCHITECT_SYSTEM_PROMPT.replace(
            "{extracted_pdf_and_website_text}", self.extracted_context
        )
        self.messages.append(SystemMessage(content=formatted_system))

    def _clean_json_response(self, raw_text: str) -> Dict[str, Any]:
        """Strips markdown fences and parses raw JSON safely."""
        text = raw_text.strip()
        # Remove markdown code blocks if the model wrapped output
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

        # Try to locate the JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            logging.warning("[Master Architect] JSON parse error: %s (Raw: %r)", err, text[:200])
            # Fallback structure
            return {
                "internal_analysis": {
                    "current_turn_count": self.turn_count,
                    "parameters_covered": ["CORE_PRODUCT"],
                    "missing_parameters": [
                        "TARGET_AUDIENCE", "BOT_OBJECTIVE", "TONE_VOICE",
                        "GUARDRAILS", "KNOWLEDGE_BOUNDARIES", "HANDOFF_TRIGGERS",
                        "DATA_PRIVACY", "FALLBACK_STRATEGY", "GREETING_CLOSING"
                    ],
                    "next_target_vector": "TARGET_AUDIENCE",
                },
                "stream_question": text,
                "interview_status": "IN_PROGRESS" if self.turn_count < 8 else "COMPLETE",
            }

    async def start(self) -> Dict[str, Any]:
        """Triggers the opening turn of the interview."""
        self.turn_count = 1
        prompt = (
            f"The business owner is starting the onboarding interview. "
            f"Business Name: '{self.business_name}'. Persona Name: '{self.persona_name}'. "
            f"STRICT INSTRUCTIONS: Do NOT greet the user or say 'Hello' or 'Welcome to the Business Bot Architect' or any conversational greeting. "
            f"Start immediately by summarizing their business and products from the ingested context in exactly 1-2 sentences. "
            f"Then ask ONLY ONE question formatted strictly with a bold Markdown header on its own line: '**Question 1: Target Audience**' (or the first missing vector). "
            f"No conversational filler text."
        )
        self.messages.append(HumanMessage(content=prompt))

        logging.info("[Master Architect] Starting interview for '%s'...", self.business_name)
        response = await self.llm.ainvoke(self.messages)
        content = response.content if hasattr(response, "content") else str(response)
        self.messages.append(AIMessage(content=content))

        parsed = self._clean_json_response(content)
        parsed["turn_count"] = self.turn_count
        return parsed

    async def step(self, user_answer: str) -> Dict[str, Any]:
        """Processes the business owner's response and yields the next question or completion."""
        self.turn_count += 1
        self.messages.append(HumanMessage(content=user_answer))

        logging.info(
            "[Master Architect] Processing turn %d for '%s' | Input length: %d chars",
            self.turn_count,
            self.business_name,
            len(user_answer),
        )

        response = await self.llm.ainvoke(self.messages)
        content = response.content if hasattr(response, "content") else str(response)
        self.messages.append(AIMessage(content=content))

        parsed = self._clean_json_response(content)
        parsed["turn_count"] = self.turn_count

        # Check for completion (Target 3-4 turns, max 5 turns)
        status = str(parsed.get("interview_status", "IN_PROGRESS")).upper()
        if "COMPLETE" in status or self.turn_count >= 5:
            self.is_complete = True
            parsed["interview_status"] = "COMPLETE"
            self.finalized_blueprint = parsed.get("finalized_blueprint") or {
                "business_name": self.business_name,
                "persona_name": self.persona_name,
                "parameters_covered": parsed.get("internal_analysis", {}).get("parameters_covered", ["CORE_PRODUCT", "TARGET_AUDIENCE", "BOT_OBJECTIVE", "TONE_VOICE", "GUARDRAILS"]),
                "summary": "Completed focused onboarding architect interview.",
            }
            parsed["finalized_blueprint"] = self.finalized_blueprint
            logging.info("[Master Architect] Interview COMPLETE for '%s' at turn %d", self.business_name, self.turn_count)

        return parsed

    async def finish(self) -> Dict[str, Any]:
        """Manually finalizes the interview without answering more questions."""
        self.is_complete = True
        self.finalized_blueprint = {
            "business_name": self.business_name,
            "persona_name": self.persona_name,
            "parameters_covered": [
                "CORE_PRODUCT", "TARGET_AUDIENCE", "BOT_OBJECTIVE", "TONE_VOICE",
                "GUARDRAILS", "KNOWLEDGE_BOUNDARIES", "HANDOFF_TRIGGERS",
                "DATA_PRIVACY", "FALLBACK_STRATEGY", "GREETING_CLOSING"
            ],
            "summary": "Interview completed with current inputs.",
        }
        return {
            "interview_status": "COMPLETE",
            "turn_count": self.turn_count,
            "finalized_blueprint": self.finalized_blueprint,
            "stream_question": "",
        }
