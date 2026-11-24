import json
import re
import time
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from business_assistant.prompts import MASTER_ARCHITECT_SYSTEM_PROMPT, get_azure_or_fallback_llm
from src.logger import logging, log_latency


ALL_VECTORS = [
    "CORE_PRODUCT",
    "TARGET_AUDIENCE",
    "BOT_OBJECTIVE",
    "TONE_VOICE",
    "GUARDRAILS",
    "KNOWLEDGE_BOUNDARIES",
    "HANDOFF_TRIGGERS",
    "DATA_PRIVACY",
    "FALLBACK_STRATEGY",
    "GREETING_CLOSING",
]

DEFAULT_FALLBACK_RULES = {
    "GUARDRAILS": "Strictly quote brochure prices; NEVER offer unlisted discounts or custom guarantees.",
    "KNOWLEDGE_BOUNDARIES": "If queried about topics, products, or services outside verified brochure and website materials, politely decline to answer, state that this is outside your domain, and invite the customer to share their name and contact details (email or phone) so the team can assist them directly.",
    "HANDOFF_TRIGGERS": "Whenever a customer expresses purchase intent, booking requests, or asks for a callback/quote, invite them to share their name and email or phone number for the team to contact them.",
    "DATA_PRIVACY": "Never request or record credit card numbers or passwords. When a customer provides their name, email, or phone number, silently capture it for the admin's Inquiries & Leads dashboard and politely acknowledge receipt (e.g., 'Thank you, our team has noted your details and will reach out shortly'), but NEVER repeat or echo their phone number or email back in the chat.",
    "FALLBACK_STRATEGY": "If unable to resolve a customer inquiry after two attempts or if the answer is not in the knowledge base, politely apologize, state the boundary, and offer to have a team member follow up if they provide their contact details.",
    "GREETING_CLOSING": "Deliver a welcoming greeting on turn 1 only; polite sign-off when the customer concludes with thanks or goodbye.",
}


class MasterArchitectInterview:
    """
    Adaptive State Machine that conducts the live WebSocket interview with the business owner,
    probing across 10 business vectors with a maximum of 10 substantive questions,
    distinguishing clarifications from new questions, and outputting truthful blueprints.
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
        self.question_count = 0
        self.clarification_count = 0
        self.messages: List[Any] = []
        self.vector_states: Dict[str, Dict[str, Any]] = {}
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
            # Secondary defensive attempt: strip trailing commas and unescaped control whitespace
            try:
                cleaned = re.sub(r",\s*([\]}])", r"\1", text)
                cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", lambda m: "\n" if m.group(0) == "\n" else " ", cleaned)
                return json.loads(cleaned)
            except Exception:
                pass

            logging.warning("[Master Architect] JSON parse error: %s (Raw: %r)", err, text[:200])
            # Fallback structure preserving counters
            return {
                "internal_analysis": {
                    "question_count": self.question_count,
                    "is_clarification": False,
                    "parameters_covered": list(self.vector_states.keys()) or ["CORE_PRODUCT"],
                    "missing_parameters": [v for v in ALL_VECTORS if v not in self.vector_states],
                    "next_target_vector": "TARGET_AUDIENCE",
                },
                "stream_question": text,
                "interview_status": "IN_PROGRESS" if self.question_count < 10 else "COMPLETE",
            }

    def _update_vector_states(self, parsed: Dict[str, Any], source_label: str = "owner") -> None:
        """Updates internal vector states truthfully from model extraction."""
        analysis = parsed.get("internal_analysis", {})
        details = analysis.get("vector_details", {})

        # 1. Ingest detailed vector states if provided
        if isinstance(details, dict):
            for vec_id, vdata in details.items():
                if vec_id in ALL_VECTORS and isinstance(vdata, dict):
                    self.vector_states[vec_id] = {
                        "state": vdata.get("state", "VERIFIED_OWNER"),
                        "value": vdata.get("value", ""),
                        "source": vdata.get("source", source_label),
                    }

        # 2. Ingest parameters_covered array defensively
        covered = analysis.get("parameters_covered", [])
        if isinstance(covered, list):
            for vec_id in covered:
                if vec_id in ALL_VECTORS and vec_id not in self.vector_states:
                    self.vector_states[vec_id] = {
                        "state": "VERIFIED_OWNER",
                        "value": f"Confirmed during turn {self.turn_count}",
                        "source": source_label,
                    }

    def _compile_blueprint(self, parsed: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compiles a truthful blueprint distinguishing verified attributes from unresolved fallback rules."""
        verified = {}
        unresolved = []
        safe_rules = {}

        for vec_id in ALL_VECTORS:
            if vec_id in self.vector_states and self.vector_states[vec_id].get("state") in ("VERIFIED_OWNER", "VERIFIED_CONTEXT"):
                verified[vec_id] = self.vector_states[vec_id]
            else:
                unresolved.append(vec_id)
                if vec_id in DEFAULT_FALLBACK_RULES:
                    safe_rules[vec_id] = DEFAULT_FALLBACK_RULES[vec_id]

        blueprint = {
            "business_name": self.business_name,
            "persona_name": self.persona_name,
            "verified_vectors": verified,
            "unresolved_vectors": unresolved,
            "safe_fallback_rules": safe_rules,
            "parameters_covered": list(verified.keys()),
            "total_questions_asked": self.question_count,
            "total_turns": self.turn_count,
            "summary": f"Onboarding interview completed with {len(verified)}/{len(ALL_VECTORS)} verified business vectors.",
        }

        # Merge any extra high-level summary from model if present
        if parsed and isinstance(parsed.get("finalized_blueprint"), dict):
            model_bp = parsed["finalized_blueprint"]
            if "summary" in model_bp:
                blueprint["summary"] = model_bp["summary"]

        return blueprint

    @traceable(name="MasterArchitect.start_turn", run_type="chain")
    async def start(self) -> Dict[str, Any]:
        """Triggers the opening turn of the interview (Question 1)."""
        self.turn_count = 1
        self.question_count = 1
        self.clarification_count = 0

        has_rich_context = bool(self.extracted_context and len(self.extracted_context.strip()) >= 150 and "No website or brochure" not in self.extracted_context)
        if has_rich_context:
            context_instruction = (
                "Start immediately by summarizing their business and products from the ingested context in exactly 1-2 sentences. "
                "Then ask ONLY ONE question formatted strictly with a bold Markdown header on its own line: '**Question 1: Target Audience**' (or the first missing vector)."
            )
        else:
            context_instruction = (
                "The ingested business materials are minimal or unavailable. In 1 concise sentence, state that you are ready to construct their custom business assistant. "
                "Then ask ONLY ONE question formatted strictly with a bold Markdown header on its own line: '**Question 1: Core Offerings & Audience**', asking them to describe their 2-3 main products/services and primary customer demographic."
            )

        prompt = (
            f"The business owner is starting the onboarding interview. "
            f"Business Name: '{self.business_name}'. Persona Name: '{self.persona_name}'. "
            f"STRICT INSTRUCTIONS: Do NOT greet the user or say 'Hello' or 'Welcome to the Business Bot Architect' or any conversational greeting. "
            f"{context_instruction} "
            f"No conversational filler text."
        )
        self.messages.append(HumanMessage(content=prompt))

        logging.info("[Master Architect] Starting interview for '%s'...", self.business_name)
        t_arch0 = time.perf_counter()
        run_config = {
            "run_name": f"MasterArchitect.start: {self.business_name}",
            "tags": ["master_architect", "interview", f"biz:{self.business_name}"],
            "metadata": {"business_name": self.business_name, "turn": self.turn_count},
        }
        response = await self.llm.ainvoke(self.messages, config=run_config)
        arch_ms = (time.perf_counter() - t_arch0) * 1000.0
        log_latency("ARCHITECT_INTERVIEW", f"Biz: '{self.business_name}' | Turn 1 (start) | Latency: {arch_ms:.1f}ms")

        content = response.content if hasattr(response, "content") else str(response)
        self.messages.append(AIMessage(content=content))

        parsed = self._clean_json_response(content)
        self._update_vector_states(parsed, source_label="context")

        parsed["turn_count"] = self.turn_count
        parsed["question_count"] = self.question_count
        parsed["is_clarification"] = False
        return parsed

    @traceable(name="MasterArchitect.step_turn", run_type="chain")
    async def step(self, user_answer: str) -> Dict[str, Any]:
        """Processes the business owner's response and yields the next question or completion."""
        self.turn_count += 1
        self.messages.append(HumanMessage(content=user_answer))

        logging.info(
            "[Master Architect] Processing turn %d (Q%d) for '%s' | Input length: %d chars",
            self.turn_count,
            self.question_count,
            self.business_name,
            len(user_answer),
        )

        t_step0 = time.perf_counter()
        run_config = {
            "run_name": f"MasterArchitect.step (Q{self.question_count}): {self.business_name}",
            "tags": ["master_architect", "interview", f"biz:{self.business_name}"],
            "metadata": {"business_name": self.business_name, "turn": self.turn_count, "question": self.question_count},
        }
        response = await self.llm.ainvoke(self.messages, config=run_config)
        step_ms = (time.perf_counter() - t_step0) * 1000.0
        log_latency("ARCHITECT_INTERVIEW", f"Biz: '{self.business_name}' | Turn {self.turn_count} (Q{self.question_count}) | Latency: {step_ms:.1f}ms")

        content = response.content if hasattr(response, "content") else str(response)
        self.messages.append(AIMessage(content=content))

        parsed = self._clean_json_response(content)

        # Check if model flagged this turn as a clarification
        analysis = parsed.get("internal_analysis", {})
        is_clarif = bool(analysis.get("is_clarification", False))
        if is_clarif:
            self.clarification_count += 1
        else:
            self.question_count += 1

        self._update_vector_states(parsed, source_label="owner")

        # Check completion conditions:
        # 1. Model set interview_status to COMPLETE
        # 2. Reached absolute maximum of 10 substantive questions
        # 3. All essential vectors covered
        status = str(parsed.get("interview_status", "IN_PROGRESS")).upper()
        if "COMPLETE" in status or self.question_count >= 10:
            self.is_complete = True
            parsed["interview_status"] = "COMPLETE"
            self.finalized_blueprint = self._compile_blueprint(parsed)
            parsed["finalized_blueprint"] = self.finalized_blueprint
            parsed["stream_question"] = ""
            logging.info(
                "[Master Architect] Interview COMPLETE for '%s' at turn %d (Questions: %d, Clarifications: %d, Verified: %d)",
                self.business_name,
                self.turn_count,
                self.question_count,
                self.clarification_count,
                len(self.finalized_blueprint["verified_vectors"]),
            )

        parsed["turn_count"] = self.turn_count
        parsed["question_count"] = min(self.question_count, 10)
        parsed["is_clarification"] = is_clarif
        return parsed

    async def finish(self) -> Dict[str, Any]:
        """
        Manually finalizes the interview when the user finishes early.
        Truthfully preserves verified attributes and applies safe fallback rules for unresolved vectors.
        """
        self.is_complete = True
        self.finalized_blueprint = self._compile_blueprint()
        logging.info(
            "[Master Architect] Finish Early invoked for '%s' at turn %d. Verified: %d, Unresolved: %d",
            self.business_name,
            self.turn_count,
            len(self.finalized_blueprint["verified_vectors"]),
            len(self.finalized_blueprint["unresolved_vectors"]),
        )
        return {
            "interview_status": "COMPLETE",
            "turn_count": self.turn_count,
            "question_count": min(self.question_count, 10),
            "finalized_blueprint": self.finalized_blueprint,
            "stream_question": "",
        }
