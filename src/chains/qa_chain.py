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
    citations = set()
    seen = set()

    for doc in docs:
        source_label = _format_source_label(doc.metadata)
        locator = _format_locator(doc.metadata)
        section = _format_section(doc.metadata)
        citation = f"{source_label} - {locator}"
        if section:
            citation = f"{citation} - {section}"
        if citation in seen:
            continue
        seen.add(citation)
        citations.add(citation)
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
        label = _format_source_label(doc.metadata)
        if label not in seen:
            seen.add(label)
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

    lowered = text.lower()
    banned_fragments = [
        "there is nothing in this chunk",
        "the provided chunks do not contain",
        "based on the provided context",
        "the context does not mention",
        "the provided context does not",
        "i cannot find the answer to that in the provided documents",
        "retrieved context",
    ]
    if text == NOT_FOUND_TOKEN or any(fragment in lowered for fragment in banned_fragments):
        return FALLBACK_ANSWER

    hedge_fragments = [
        "it appears that",
        "it appears",
        "seems to",
        "seems like",
        "suggests that",
        "suggesting that",
        "implying that",
        "could be interpreted",
        "may be related",
        "possibly related",
        "without further context",
        "without more context",
        "it's difficult to",
        "it is difficult to",
        "unfortunately, i couldn't",
        "unfortunately, without",
        "if you have any additional details",
        "i'll do my best to assist",
    ]
    hedge_count = sum(1 for fragment in hedge_fragments if fragment in lowered)
    if hedge_count >= 2:
        return FALLBACK_ANSWER

    text = re.sub(r"(?i)^based on the provided context[:,]?\s*", "", text).strip()
    text = re.sub(r"(?i)^according to the provided context[:,]?\s*", "", text).strip()
    return text or FALLBACK_ANSWER


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are DocMind, a professional assistant for answering questions from uploaded materials.

Rules:
1. Use only the supplied source notes.
2. Do not add facts that are not supported by those notes.
3. Never mention internal implementation details such as chunks, embeddings, vector search, retrieval, source passages, or context windows.
4. If the answer is not clearly supported, reply with exactly {NOT_FOUND_TOKEN}
5. Write naturally, clearly, and professionally.
6. Structure the answer when helpful, and keep it readable — but do not shorten an answer just for brevity's sake.
7. Do not add a citation section yourself.
8. Never hedge or speculate. Do not use phrases like "it appears," "seems to," "suggests," "implying," "may be," "possibly," "without further context," or "it's difficult to say." If you are not confident enough to state something directly, that specific point does not belong in the answer at all — omit it rather than softening it.
9. State facts directly and plainly, as if you simply know them from the material. Do not narrate your own uncertainty at any point in the answer.
10. If only partial information is available, answer confidently with the part that is clearly supported, and say nothing about the part that isn't — do not apologize for incompleteness or ask the user for clarification.
11. If the retrieved material fully and directly answers the question, give a thorough, complete explanation using all the relevant information available — do not compress a well-supported answer into a short summary. Only keep an answer brief when the source material itself is genuinely limited.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """Uploaded material:
{sources}

Question: {question}

Answer directly and confidently, using only what's clearly supported above. If the material fully covers the question, be thorough and complete.""",
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
        api_key=os.environ["GROQ_API_KEY"],
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
        )

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