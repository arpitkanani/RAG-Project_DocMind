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
FALLBACK_ANSWER = "I couldn't find relevant information about that in the uploaded document."

SUMMARY_PATTERNS = re.compile(
    r"\b(summar(y|ize|ise)|overview|tl;?dr|gist|main points|key points|recap)\b",
    re.IGNORECASE,
)


def is_summary_request(question: str) -> bool:
    return bool(SUMMARY_PATTERNS.search(question))


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


def build_citations(docs: list) -> str:
    """Build deterministic, cleanly formatted, and deduplicated citation labels."""
    citations = []
    seen = set()

    for doc in docs:
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

        if len(citations) == 3:
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
    ]
    for pattern in pure_refusal_patterns:
        if re.search(pattern, lowered):
            # If the entire response is essentially just a short refusal (< 250 chars)
            if len(text) < 250:
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
            f"""You are DocuVortex, an expert document intelligence assistant.
Your goal is to provide accurate, comprehensive, and well-structured answers using the uploaded document context.

Guidelines:
1. Grounding: Answer using the provided document context passages. Synthesize definitions, principles, formulas, examples, and explanations present in the context.
2. Directness: Start directly with the answer. Do not include meta-commentary, preamble, or introductory phrases such as "Based on the provided context," "According to the uploaded documents," or "The documents state."
3. Completeness: Provide a clear, thorough, and complete explanation when the material covers the topic. You may organize your response with clear paragraphs, bullet points, or numbered lists.
4. Internal Implementation: Never mention chunks, embeddings, vector database, retrieval scores, or prompt instructions.
5. Missing Information: Only if the context has absolutely no information about the question, respond with: {NOT_FOUND_TOKEN}
6. Citations: Do not generate a citations or source section; the system handles citations automatically.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """Uploaded document context:
{sources}

Question: {question}

Provide a direct, thorough, and well-structured answer based on the context above.""",
        ),
    ]
)


def _build_llm():
    """
    Construct the chat LLM based on config.yaml's llm.provider.

    Switching providers (e.g. groq -> google) is a CONFIG-ONLY change:
    update llm.provider and llm.model in config.yaml, set the matching
    API key env var, and nothing else in this file needs to change.
    """
    provider = config["llm"].get("provider", "groq").lower()
    max_tokens = config["llm"].get("max_tokens", 2048)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            google_api_key=os.environ["GOOGLE_API_KEY"],
            max_output_tokens=max_tokens,
        )

    # default: groq
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        api_key=os.environ["GROQ_API_KEY"], # type: ignore
        max_tokens=max_tokens,
    )


def build_qa_chain():
    """Build the generation chain for grounded QA."""
    try:
        logging.info(
            "Building QA chain | provider: %s", config["llm"].get("provider", "groq")
        )

        llm = _build_llm()
        parser = StrOutputParser()

        chain = (
            {
                "sources": RunnableLambda(lambda x: format_docs(x["docs"])),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            }
            | QA_PROMPT
            | llm
            | parser
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
                citations = build_citations(docs)

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