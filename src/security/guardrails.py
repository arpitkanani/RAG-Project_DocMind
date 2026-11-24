"""DocuVortex Security & Anti-Leak Guardrails Engine.

Provides deterministic zero-latency guardrail checks to protect:
1. Model identity and provider secrecy (Groq, Gemini, Azure OpenAI, etc.).
2. Internal backend and application source code (Python implementation, StateGraph, DB).
3. System prompts, developer instructions, and jailbreak attempts.
4. System credentials, API keys, and environment secrets.

Aligned with the business bot security standards in business_assistant/prompts.py.
"""

import re
from typing import Optional, Tuple

MODEL_REFUSAL_RESPONSE = (
    "I am **DocuVortex**, an intelligent AI assistant powered by DocuVortex's "
    "proprietary agentic document intelligence architecture. Specific internal model "
    "weights, multi-tier provider configurations, and underlying infrastructure details "
    "are kept confidential. How can I assist you with your documents or research today?"
)

CODE_REFUSAL_RESPONSE = (
    "I am here exclusively to help you analyze your uploaded documents, explore data, "
    "and answer research questions. I cannot share, display, or disclose internal backend "
    "code, application source code, or proprietary operational implementations."
)

PROMPT_REFUSAL_RESPONSE = (
    "I cannot share, summarize, or disclose internal system prompts, developer "
    "instructions, or operational rules. I am here to help you safely explore your "
    "documents and answer your questions."
)

CREDENTIAL_REFUSAL_RESPONSE = (
    "Security and privacy policies strictly prohibit sharing system credentials, "
    "API keys, database connection strings, or environment secrets."
)


# 1. Model Identity & Provider Probing Patterns
MODEL_PROBE_PATTERNS = [
    re.compile(
        r"\b(?:which|what)\s+(?:ai\s+|language\s+|underlying\s+|base\s+)?(?:model|llm)\s+"
        r"(?:are\s+you(?:\s+using)?|is\s+(?:this|being\s+used|answering|powering|running)|do\s+you\s+use|we\s+are\s+using)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+(?:ai\s+|language\s+|underlying\s+|base\s+)?(?:model|llm)\s+"
        r"(?:powers?|runs?|answers?|generates?)\s+(?:this|you|responses?|answers?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+(?:model|llm)\s+(?:do\s+you|are\s+you|is\s+used|is\s+this|powering)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhich\s+model\s+(?:are\s+we\s+using|are\s+you\s+using|is\s+used)\s+to\s+answer\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:model|llm)\s+are\s+we\s+using\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:are\s+you|is\s+this)\s+(?:based\s+on\s+|powered\s+by\s+)?(?:chatgpt|gpt-?3(?:\.5)?|gpt-?4|gpt-?5|gemini|groq|claude|llama|qwen|mistral|deepseek|openai)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+(?:llm|foundation\s+model)\s+(?:powers?|runs?|is\s+behind)\s+(?:you|this|docuvortex)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:tell\s+me|name)\s+(?:the|your)\s+(?:exact\s+)?(?:model|llm)\s+(?:name|version|architecture)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*what\s+(?:is\s+your\s+model|model\s+are\s+you)\s*[\?\!\.]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:which|what)\s+model\s+(?:version\s+)?is\s+generating\s+(?:this|the\s+answers?|the\s+responses?|responses?|answers?)\b",
        re.IGNORECASE,
    ),
]

# 2. Source Code & Backend Implementation Probing Patterns
CODE_PROBE_PATTERNS = [
    re.compile(
        r"\b(?:show|give|display|print|share|reveal|view|send|dump|provide)\s+(?:me\s+)?(?:the\s+)?"
        r"(?:source\s+)?code\s+of\s+(?:our|this|your|the)\s+(?:system|app|application|platform|bot|website|project|workspace|backend|server|graph)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|give|display|print|share|reveal|send|provide)\s+(?:me\s+)?"
        r"(?:your|the\s+platform'?s?|the\s+app'?s?|our)\s+(?:backend|source|python|internal)\s+code\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what\s+is|show\s+me|give\s+me|share)\s+(?:the\s+)?(?:backend|source)\s+code\s+(?:of|for|behind)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:code\s+of\s+our|give\s+our\s+code|show\s+our\s+code|share\s+our\s+code)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|give|display|share)\s+(?:me\s+)?(?:your|our)\s+(?:github|repo|codebase|repository|internal\s+code)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:print|show|read|display)\s+(?:me\s+)?(?:the\s+)?(?:file\s+)?"
        r"(?:app\.py|main\.py|builder\.py|nodes_agentic\.py|nodes\.py|qa_chain\.py|vector_store\.py|config\.yaml)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:how\s+is\s+this\s+(?:backend|application|system|bot)\s+coded|give\s+me\s+the\s+code\s+of\s+this\s+(?:bot|rag|system))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:show|give\s+me|send\s+me)\s+(?:the\s+)?(?:code|source\s+code|python\s+code)\s*[\?\!\.]*\s*$",
        re.IGNORECASE,
    ),
]

# 3. System Prompt & Instruction Extraction / Jailbreak Patterns
PROMPT_LEAK_PATTERNS = [
    re.compile(
        r"\b(?:show|reveal|display|print|output|repeat|give|share)\s+(?:me\s+)?(?:the|your|our)?\s*"
        r"(?:system\s+prompt|developer\s+prompt|hidden\s+prompt|initial\s+prompt|system\s+instructions?|developer\s+instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what\s+are\s+your|what\s+is\s+your)\s+(?:system\s+prompt|system\s+instructions?|developer\s+instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|disregard\s+(?:all\s+)?prior\s+rules|"
        r"you\s+are\s+now\s+in\s+developer\s+mode|dan\s+mode|unrestricted\s+mode|jailbreak)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:repeat|print|output)\s+(?:the\s+words?\s+)?(?:above|everything\s+above|your\s+preceding\s+prompt)\b",
        re.IGNORECASE,
    ),
]

# 4. Credential & Environment Secret Probing Patterns
CREDENTIAL_PROBE_PATTERNS = [
    re.compile(
        r"\b(?:what\s+is|show\s+me|give\s+me|print|reveal|output)\s+(?:the|your|our)?\s*"
        r"(?:(?:groq|gemini|azure|openai|qdrant|supabase|service|secret|api)\s+)*(?:api[_-]?key|secret[_-]?key|api\s*key|secret\s*key|database\s+password|database\s+credentials?|env\s+file|\.env)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:groq|gemini|azure|openai|qdrant|supabase)\s+(?:api\s+)?(?:key|secret|token)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:qdrant[_-]?api[_-]?key|supabase[_-]?service[_-]?key|service[_-]?role[_-]?key)\b",
        re.IGNORECASE,
    ),
]

# Exclusions: If query is explicitly asking about code *inside* the uploaded document
DOC_CODE_EXCEPTION_PATTERNS = [
    re.compile(r"\b(?:in|from|inside)\s+(?:this|the)\s+(?:document|pdf|file|paper|section|page|chapter)\b", re.IGNORECASE),
    re.compile(r"\b(?:document|pdf|file|paper)\s+(?:contains|has|mentions)\b", re.IGNORECASE),
    re.compile(r"\bpage\s+\d+\b", re.IGNORECASE),
]


def check_security_guardrails(
    query: str,
    has_doc_context: bool = False,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Evaluates whether user query triggers security, anti-leak, or identity guardrails.

    Returns:
        (is_triggered: bool, refusal_response: Optional[str], category: Optional[str])
    """
    q = (query or "").strip()
    if not q:
        return False, None, None

    # 1. Check Model Identity Probes
    for pat in MODEL_PROBE_PATTERNS:
        if pat.search(q):
            return True, MODEL_REFUSAL_RESPONSE, "model_probe"

    # 2. Check System Prompt Extraction / Jailbreak Probes
    for pat in PROMPT_LEAK_PATTERNS:
        if pat.search(q):
            return True, PROMPT_REFUSAL_RESPONSE, "prompt_leak"

    # 3. Check Credential & Secrets Probes
    for pat in CREDENTIAL_PROBE_PATTERNS:
        if pat.search(q):
            return True, CREDENTIAL_REFUSAL_RESPONSE, "credential_probe"

    # 4. Check Code Probes (ensuring document-grounded questions are not falsely blocked)
    for pat in CODE_PROBE_PATTERNS:
        if pat.search(q):
            # If documents are active, verify if query specifically asks about code in the uploaded document
            if has_doc_context:
                is_doc_grounded = any(ex_pat.search(q) for ex_pat in DOC_CODE_EXCEPTION_PATTERNS)
                if is_doc_grounded:
                    # Legitimate document analysis question
                    continue
            return True, CODE_REFUSAL_RESPONSE, "code_probe"

    return False, None, None
