import os
import sys
from typing import List

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel
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


def format_docs(docs: list) -> str:
    """Format retrieved chunks into a structured, citation-friendly context block."""
    if not docs:
        return "No relevant content found in the indexed sources."

    formatted_docs = []
    for index, doc in enumerate(docs, start=1):
        source_label = _format_source_label(doc.metadata)
        locator = _format_locator(doc.metadata)
        collection_name = doc.metadata.get("collection_name", "unknown")
        formatted_docs.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"Source: {source_label}",
                    f"Locator: {locator}",
                    f"Collection: {collection_name}",
                    "Relevant text:",
                    doc.page_content,
                ]
            )
        )

    return "\n\n".join(formatted_docs)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are DocMind, a grounded document assistant.

Use the provided context as your primary source of truth.
You may summarize, combine, or restate information when it is clearly supported by the context, but do not invent facts that are not there.
If the answer is not supported by the context, say:
"This information is not available in the uploaded documents or video transcript."

You are a knowledgeable and helpful AI assistant. You are given a specific set of retrieved context documents and a user's question.

Your task is to answer the user's question based strictly and solely on the provided context.

Guidelines:
1. **Rely ONLY on the context**: Do not use any outside knowledge or speculate. If the answer cannot be found in the context, say exactly: "I cannot find the answer to that in the provided documents."
2. **Be direct and factual**: Keep your answers concise, accurate, and easy to understand and informative with retrived chunks or given informations of context.
3. **Cite sources**: Where possible, reference the specific document, section, or source name provided in the context.
4. **Tone**: Maintain a professional, conversational, and helpful tone. Do not mention the word "context" in your response.
5. **Structure gently**: First mentally organize the retrieved information into the most relevant facts, sections, or steps before answering.


How to answer:
- Match the user's language and be natural.
- For factual questions, answer directly first.
- Use a short paragraph, then bullet points, short sections, or numbered steps when that makes the answer clearer.
- Do not say phrases like "the transcript says", "the context says", or "based on the provided context" unless the user asks about the source wording.
- For YouTube transcript questions such as "what is this about", "summarize this video", or "main topic", give a short grounded summary if the transcript supports it.
- For resume or profile questions such as "what projects were done", prioritize project sections over certificates, skills, or unrelated metadata when the retrieved context includes a project list.
- When the retrieved details are scattered, combine them into a refined structured answer instead of copying one chunk at a time.
- If the user asks for details, extract the relevant details completely and organize them cleanly.
- Add a brief citation line at the end using filename or "YouTube transcript", plus page, row, or timestamp when available.
- If multiple sources support the answer, cite each one briefly.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer naturally and structure the response in the clearest useful way for the question.""",
        ),
    ]
)


def build_qa_chain(collection_names: List[str] | None = None):
    """Build the LCEL RAG chain for one or more collections."""
    try:
        logging.info("Building QA chain")

        llm = ChatOllama(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
        )
        parser = StrOutputParser()

        retriever_obj = Retriever(collection_names=collection_names)

        parallel_step = RunnableParallel(
            {
                "context": (
                    RunnableLambda(lambda x: x["question"])
                    | RunnableLambda(retriever_obj.retrieve)
                    | RunnableLambda(format_docs)
                ),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x["chat_history"]),
            }
        )

        chain = parallel_step | QA_PROMPT | llm | parser

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
) -> str:
    """Run retrieval, generate an answer, and persist the chat history."""
    try:
        logging.info("Processing question: %s...", question[:50])

        memory = MemoryManager(session_id=session_id)
        chat_history = memory.get_history()

        chain = build_qa_chain(collection_names)
        answer = chain.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        memory.save_message("human", question)
        memory.save_message("ai", answer)

        logging.info("Answer generated and saved")
        return answer
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)
