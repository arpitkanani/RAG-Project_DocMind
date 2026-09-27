import os
import re
import sys
from typing import List

import yaml
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from src.components.memory_manager import MemoryManager
from src.components.retriever import Retriever
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging
from src.utils.rate_limiter import LLMRateLimitError, raise_as_rate_limit_error, llm_rate_limiter

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

NOT_FOUND_TOKEN = "DATA_NOT_FOUND"
FALLBACK_ANSWER = "I couldn't find information about that in the uploaded document(s)."

DOC_SUMMARY_PATTERNS = re.compile(
    r"^\s*(can you\s+)?(please\s+)?(give\s+(me\s+)?(a\s+)?)?"
    r"(summar(y|ize|ise)|overview|tl;?dr|recap|gist|main points|key takeaways)"
    r"(\s+(of\s+)?(this\s+|the\s+)?(document|file|pdf|paper|text|book|upload|manual|material|doc|whole thing|entire document))?"
    r"\s*[\?\!\.]*$",
    re.IGNORECASE,
)


def is_summary_request(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False

    # Targeted topic queries (even with the word "summary") MUST use semantic vector search (retrieve_qa)
    specific_topic_words = {
        "chart", "gantt", "cost", "price", "budget", "table", "figure", "section", "chapter",
        "requirement", "requirements", "author", "name", "architecture", "method", "results",
        "conclusion", "schedule", "timeline", "estimate", "risk", "diagram", "code", "implementation",
    }
    tokens = set(re.findall(r"\b[a-z0-9]+\b", q))
    if tokens & specific_topic_words:
        return False

    # Check for general whole-document summary patterns
    if DOC_SUMMARY_PATTERNS.match(q):
        return True

    # Questions like "what is this document about", "what is this file about"
    if re.search(r"\bwhat\s+(is\s+)?(this\s+|the\s+)?(document|file|pdf|paper)\s+about\b", q):
        return True

    # Standalone words like "summary", "summarize", "overview", "tldr"
    if q in {"summary", "summarize", "summarise", "overview", "tl;dr", "tldr", "recap", "gist"}:
        return True

    return False


def _format_source_label(metadata: dict) -> str:
    source = metadata.get("source", "Unknown source")
    source_type = metadata.get("type")

    if source_type == "youtube":
        return "YouTube transcript"

    return os.path.basename(str(source)) or "Unknown source"


def _format_locator(metadata: dict) -> str:
    if metadata.get("page") is not None:
        try:
            return f"page {int(metadata['page']) + 1}"
        except (TypeError, ValueError):
            return f"page {metadata['page']}"

    if metadata.get("timestamp"):
        return f"timestamp {metadata['timestamp']}"

    if metadata.get("row") is not None:
        return f"row {metadata['row']}"

    return "location unknown"


def _format_section(metadata: dict) -> str:
    for key in ["section", "heading", "title", "sheet_name"]:
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def format_docs(docs: list) -> str:
    """Format retrieved passages into a structured source block for generation."""
    if not docs:
        return "No grounded source passages are available."

    formatted_docs = []
    for index, doc in enumerate(docs, start=1):
        source_label = _format_source_label(doc.metadata)
        locator = _format_locator(doc.metadata)
        section = _format_section(doc.metadata)
        lines = [
            f"Source Note {index}",
            f"Document: {source_label}",
            f"Location: {locator}",
        ]
        if section:
            lines.append(f"Section: {section}")
        lines.extend(["Passage:", doc.page_content])
        formatted_docs.append("\n".join(lines))

    return "\n\n".join(formatted_docs)


def build_citations(docs: list, answer: str = "") -> str:
    """Build deterministic, cleanly formatted citations ONLY for documents
    whose content was actually referenced and used in the synthesized answer."""
    if not docs:
        return ""

    citations = []
    seen = set()

    # Smart filtering: only include docs whose terms or concepts are used in the answer
    used_docs = []
    if answer and answer.strip() and answer != FALLBACK_ANSWER:
        # Extract meaningful tokens (lowercase words >= 4 chars, excluding stopwords)
        answer_tokens = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", answer.lower()))
        common_words = {
            "that", "with", "this", "from", "they", "were", "been", "have", "their", "which",
            "about", "would", "there", "these", "could", "other", "into", "more", "first",
            "than", "them", "some", "what", "when", "where", "also", "only", "such", "after",
            "project", "document", "report", "system", "please", "using", "based"
        }
        distinctive_tokens = answer_tokens - common_words

        for doc in docs:
            content = (doc.page_content or "").lower()
            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", content))
            overlap = distinctive_tokens & doc_tokens
            if len(overlap) >= 2 or len(docs) == 1:
                used_docs.append(doc)

        if not used_docs:
            return ""
    else:
        used_docs = docs[:1]

    for doc in used_docs:
        if not getattr(doc, "metadata", None):
            continue
        source_label = _format_source_label(doc.metadata)
        locator = _format_locator(doc.metadata)
        section = _format_section(doc.metadata)

        # Build clean citation string
        citation_parts = [source_label]
        if locator and locator != "location unknown":
            citation_parts.append(locator)
        if section:
            citation_parts.append(section)

        citation = " - ".join(citation_parts)

        # Deduplication check
        canonical_key = (source_label.lower(), locator.lower(), section.lower())
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        citations.append(citation)

        # For focused answers, cap at 2 max (1 is standard for single-location answers)
        if len(citations) >= 2:
            break

    if not citations:
        return ""

    return "Source:\n" + "\n".join(citations)


def build_source_only_citations(docs: list) -> str:
    """One citation per unique SOURCE only (no locator) -- used for summary
    answers, where listing every page/timestamp would be noise, not signal."""
    seen = set()
    labels = []
    for doc in docs:
        if not getattr(doc, "metadata", None):
            continue
        label = _format_source_label(doc.metadata)
        canonical = label.strip().lower()
        if canonical not in seen:
            seen.add(canonical)
            labels.append(label)
    if not labels:
        return ""
    return "Source:\n" + "\n".join(labels)


def merge_same_location_docs(docs: list) -> list:
    """
    Merge chunks that share the same source + locator (e.g. same PDF page)
    into a single Document, so the LLM never sees the same page as two
    separate 'locations' and doesn't narrate a false multi-location story.
    """
    merged: dict[tuple, Document] = {}
    order: list[tuple] = []

    for doc in docs:
        source_label = _format_source_label(doc.metadata)
        locator = _format_locator(doc.metadata)
        key = (source_label, locator)

        if key not in merged:
            merged[key] = Document(
                page_content=doc.page_content,
                metadata=dict(doc.metadata),
            )
            order.append(key)
        else:
            existing = merged[key]
            if doc.page_content not in existing.page_content:
                existing.page_content = f"{existing.page_content}\n{doc.page_content}"

    return [merged[key] for key in order]


def sanitize_answer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return FALLBACK_ANSWER

    # Strip any "Source:" or "Citations:" section the model added on its own
    text = re.split(r"(?i)\n\s*(sources?|citations?)\s*:\s*\n", text)[0].strip()

    # Exact or near-exact match for not-found token
    if text == NOT_FOUND_TOKEN or text.strip(" .!?:") == NOT_FOUND_TOKEN:
        return FALLBACK_ANSWER

    # If NOT_FOUND_TOKEN appears as the entire message or leading statement
    if text.startswith(NOT_FOUND_TOKEN):
        remaining = text[len(NOT_FOUND_TOKEN):].strip(" :-.\n")
        if not remaining or len(remaining) < 30:
            return FALLBACK_ANSWER
        text = remaining

    # Strip common leading preambles cleanly instead of discarding the whole answer
    text = re.sub(
        r"(?i)^(based on|according to)\s+(the\s+)?(provided|retrieved|uploaded|given)?\s*(context|documents?|materials?|notes?|chunks?|sources?|text)[:,]?\s*",
        "",
        text,
    ).strip()
    text = re.sub(
        r"(?i)^(from the (provided|uploaded|retrieved) (documents?|materials?|context)[:,]?\s*)",
        "",
        text,
    ).strip()

    lowered = text.lower().strip()

    # Check for pure refusals (where the entire answer is just stating it couldn't find the info)
    pure_refusal_patterns = [
        r"^i couldn'?t find (any )?relevant information",
        r"^i cannot find the answer to that in the provided",
        r"^the provided (context|documents?|chunks?|material) do(es)? not contain",
        r"^the provided (context|documents?|chunks?|material) do(es)? not (provide|mention)",
        r"^there is no (information|mention) (about|regarding|on) .* in the (provided|uploaded)",
        r"^no information (is|was) provided (about|regarding|on)",
        r"^this information is not provided in the uploaded material",
        r"^the uploaded document(s)? do(es)? not (contain|mention|provide|suggest)",
        r"^i am only (able|here) to (answer|provide)",
        r"^i do not have (any )?(information|suggestions)",
    ]
    for pattern in pure_refusal_patterns:
        if re.search(pattern, lowered):
            # Only treat as refusal if it does NOT contain actual substance or entity mentions
            substantive_clues = {"project", "name", "author", "cost", "chart", "gantt", "section", "table", "step", "page", "platform", "booking", "dezlor"}
            found_substance = any(clue in lowered for clue in substantive_clues)
            if not found_substance and len(text) < 120:
                return FALLBACK_ANSWER

    # Discard conversation meta-summaries that leak into answers (e.g., "The user asked...", "The assistant replied...", "No further action or decision was made")
    if re.search(r"(?i)\b(the user asked (the assistant|about)|the assistant replied that|no further action or decision was made|the assistant responded that)\b", lowered):
        return FALLBACK_ANSWER

    text = _dedupe_near_identical_sentences(text)
    return text or FALLBACK_ANSWER


def _dedupe_near_identical_sentences(text: str) -> str:
    """
    Generations sometimes restate the exact same point twice in one reply,
    just reworded slightly. Drops duplicate points while preserving the
    overall explanation.
    """
    connectors = {"however", "additionally", "furthermore", "also", "moreover", "therefore"}
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    kept = []
    for sentence in sentences:
        words = [w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if w not in connectors]
        normalized = " ".join(words)
        if normalized and normalized in seen:
            continue  # near-identical to a sentence already kept -- drop it
        if normalized:
            seen.add(normalized)
        kept.append(sentence)
    return " ".join(kept)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are DocuVortex, an intelligent, helpful, and highly accurate document assistant.
Your goal is to provide clear, thorough, and informative answers based strictly on the provided document context.

GUIDELINES:
1. Strictly Grounded:
   - Rely EXCLUSIVELY on the facts, concepts, details, and entities contained in the uploaded document context.
   - NEVER use external pre-trained knowledge to invent, extrapolate, or introduce external technologies, modern features, or unmentioned topics that are not present in the uploaded document context.
   - Do NOT invent hypothetical comparisons or lists of outside features (for example, if asked what is missing or what new features are not in the document, do NOT invent external modern AI tools, models, or frameworks from outside knowledge).

2. Missing or Uncovered Topics:
   - If the uploaded document context does NOT contain information about the asked concept, technology, or topic (e.g. asking about "Convolution Neural Network and Transfer Learning" or modern features not discussed in the document), reply with: {NOT_FOUND_TOKEN}
   - If the user asks what the document does or does not cover, describe only what is explicitly in the context; do not extrapolate or introduce ungrounded outside subjects.
   - If only partial information is available in the context, provide what the document explicitly covers and clarify what aspects are not mentioned.

3. Helpful & Flexible Understanding:
   - Understand synonyms, abbreviations, and related terms for entities that ARE in the context (e.g. if the user asks "what project is Arpit making" and the document lists "Project name: X" with "Author: Arpit", clearly state the project name and author).
   - When asked to summarize, explain, compare, extract key points, or analyze specific topics (such as charts, timelines, or costs) that are present in the context, synthesize the information thoroughly and clearly using the provided context.

4. Clarity & Formatting:
   - Use clean Markdown (bullet points, bold key terms, tables, or numbered lists) to make information readable.
   - Be direct: do not use empty meta-filler like "Based on the provided context," "According to the uploaded documents," or "The text states." Start directly with the answer.
   - Never output meta-dialogue commentary about the user or prior assistant responses (e.g. do not say "The user asked..." or "The assistant replied...").

5. Priority:
   - Always evaluate the freshly provided document context below independently. Even if information was not found in prior chat turns, answer directly if the new context contains the information.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """Uploaded document context:
{sources}

Question: {question}

Answer the question clearly and helpfully using the context above:""",
        ),
    ]
)


# Confirmed working free-tier models (Groq and Gemini)
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-safeguard-20b",
]

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]


def _build_llm():
    """
    Constructs the chat LLM with automatic cross-provider fallback.
    If primary model hits rate limit or error, automatically tries
    fallback models from both Gemini and Groq in priority order.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    temperature = config["llm"].get("temperature", 0.1)
    max_tokens = config["llm"].get("max_tokens", 1024)

    llm_instances = []
    primary_provider = os.getenv("LLM_PROVIDER", config["llm"].get("provider", "google")).lower()

    if primary_provider == "google" and google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        for m in GEMINI_MODELS:
            try:
                llm_instances.append(
                    ChatGoogleGenerativeAI(
                        model=m,
                        temperature=temperature,
                        google_api_key=google_api_key,
                        max_output_tokens=max_tokens,
                    )
                )
            except Exception:
                pass
        if groq_api_key:
            from langchain_groq import ChatGroq
            for m in GROQ_MODELS:
                try:
                    llm_instances.append(
                        ChatGroq(
                            model=m,
                            temperature=temperature,
                            api_key=groq_api_key,
                            groq_api_key=groq_api_key,
                            max_tokens=max_tokens,
                        )
                    )
                except Exception:
                    pass
    else:
        if groq_api_key:
            from langchain_groq import ChatGroq
            for m in GROQ_MODELS:
                try:
                    llm_instances.append(
                        ChatGroq(
                            model=m,
                            temperature=temperature,
                            api_key=groq_api_key,
                            groq_api_key=groq_api_key,
                            max_tokens=max_tokens,
                        )
                    )
                except Exception:
                    pass
        if google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            for m in GEMINI_MODELS:
                try:
                    llm_instances.append(
                        ChatGoogleGenerativeAI(
                            model=m,
                            temperature=temperature,
                            google_api_key=google_api_key,
                            max_output_tokens=max_tokens,
                        )
                    )
                except Exception:
                    pass

    if not llm_instances:
        raise ValueError("Neither GROQ_API_KEY nor GOOGLE_API_KEY is configured.")

    primary = llm_instances[0]
    if len(llm_instances) > 1:
        return primary.with_fallbacks(llm_instances[1:])
    return primary


def build_qa_chain():
    """Build the generation chain for grounded QA with extract_text normalization."""
    try:
        from src.utils.helpers import extract_text

        logging.info("Building QA chain with multi-model fallbacks")

        llm = _build_llm()

        chain = (
            {
                "sources": RunnableLambda(lambda x: format_docs(x["docs"])),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            }
            | QA_PROMPT
            | llm
            | RunnableLambda(lambda x: extract_text(getattr(x, "content", x)))
        ).with_config(run_name="qa_generation")

        logging.info("QA chain built successfully")
        return chain
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)


def get_answer(
    question: str,
    collection_names: List[str] | None = None,
    session_id: str = "default",
    user_id: str | None = None,
    message_attachments: List[dict] | None = None,
) -> str:
    """Run retrieval, generate an answer, and persist the chat history."""
    try:
        logging.info("Processing question: %s...", question[:50])

        memory = MemoryManager(session_id=session_id, user_id=user_id)
        chat_history = memory.get_history()
        retriever = Retriever(collection_names=collection_names)

        if is_summary_request(question):
            docs = retriever.get_full_context(
                max_chars=config["retriever"].get("summary_max_chars", 6000)
            )
        else:
            ranked_docs = retriever.retrieve_ranked(question)
            docs = [doc for doc, _, _ in ranked_docs]

        docs = merge_same_location_docs(docs)

        # --- manual eval logging -----------------------------------------
        # Ask a question in the live app, then come back to the log file
        # and copy these lines into your manual eval dataset's "contexts"
        # list for that question -- no re-running anything needed.
        logging.info("context: question=%r | %d chunk(s) retrieved", question, len(docs))
        for i, doc in enumerate(docs, start=1):
            logging.info("context[%d]: %s", i, doc.page_content)
        # ------------------------------------------------------------------

        memory.save_message("human", question, attachments=message_attachments)

        if not docs:
            answer = FALLBACK_ANSWER
        else:
            chain = build_qa_chain()

            # Proactive throttle -- waits for capacity BEFORE calling Groq/Google,
            # keeping us under the provider's requests-per-minute limit so most
            # 429s never happen in the first place.
            llm_rate_limiter.acquire()

            try:
                answer = chain.invoke(
                    {
                        "question": question,
                        "chat_history": chat_history,
                        "docs": docs,
                    }
                )
            except Exception as e:
                # Reactive backstop -- classifies and re-raises as
                # LLMRateLimitError with a user-friendly message if this
                # still turns out to be a rate-limit error (e.g. right
                # after a restart, or under multi-process drift). Any
                # other kind of error re-raises unchanged.
                raise_as_rate_limit_error(e)

            answer = sanitize_answer(answer)

            if is_summary_request(question):
                citations = build_source_only_citations(docs)
            else:
                citations = build_citations(docs, answer)

            if answer != FALLBACK_ANSWER and citations:
                answer = f"{answer}\n\n{citations}"

        memory.save_message("ai", answer)

        logging.info("Answer generated and saved")
        return answer
    except LLMRateLimitError:
        raise
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)