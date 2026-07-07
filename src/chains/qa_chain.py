import os
import re
import sys
from typing import List

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from src.components.memory_manager import MemoryManager
from src.components.retriever import Retriever
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

NOT_FOUND_TOKEN = "DOCMIND_NOT_FOUND"
FALLBACK_ANSWER = "I couldn't find relevant information about that in the uploaded document."


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
6. Structure the answer when helpful, but keep it concise and readable.
7. Do not add a citation section yourself and add only one citation not two times okay.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """Uploaded material:
{sources}

Question: {question}

Answer naturally and structure the response in the clearest useful way for the question.""",
        ),
    ]
)


def build_qa_chain():
    """Build the generation chain for grounded QA."""
    try:
        logging.info("Building QA chain")

        llm = ChatOllama(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
        )
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
    message_attachments: List[dict] | None = None,
) -> str:
    """Run retrieval, generate an answer, and persist the chat history."""
    try:
        logging.info("Processing question: %s...", question[:50])

        memory = MemoryManager(session_id=session_id)
        chat_history = memory.get_history()
        retriever = Retriever(collection_names=collection_names)
        ranked_docs = retriever.retrieve_ranked(question)

        memory.save_message("human", question, attachments=message_attachments)
        if not ranked_docs:
            answer = FALLBACK_ANSWER
        else:
            docs = [doc for doc, _, _ in ranked_docs]
            chain = build_qa_chain()
            answer = chain.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "docs": docs,
                }
            )
            answer = sanitize_answer(answer)
            citations = build_citations(docs)
            if answer != FALLBACK_ANSWER and citations:
                answer = f"{answer}\n\n{citations}"

        memory.save_message("ai", answer)

        logging.info("Answer generated and saved")
        return answer
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)
