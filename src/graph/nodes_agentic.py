"""Consolidated graph nodes for the DocuVortex agentic RAG workflow.

Nodes:
    classify_intent   – Detect conversational vs retrieval intent
    chitchat          – Lightweight conversational response (Gemini 3.6 Flash)
    grade_documents   – Evaluate retrieved chunk relevance (Gemini 3.6 Flash)
    fallback_response – Polite message when docs are irrelevant
"""

import os
import re
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langgraph.prebuilt import create_react_agent
from langsmith import traceable  # type: ignore

from src.chains.qa_chain import FALLBACK_ANSWER, is_summary_request
from src.components.retriever import STOP_WORDS
from src.exception import CustomException
from src.graph.helpers import extract_message_text
from src.graph.state import RAGState
from src.graph.tools import chitchat_tools
from src.logger import logging


# ─── Gemini 3.6 Flash / ReAct Model Setup ───
_gemini_flash = None


def _get_gemini_flash() -> ChatGoogleGenerativeAI:
    """Lazy-initialize and cache the Gemini Flash model."""
    global _gemini_flash
    if _gemini_flash is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        _gemini_flash = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_output_tokens=100,
        )
    return _gemini_flash


FALLBACK_MESSAGE = (
    "I couldn't find relevant information in your uploaded documents to answer "
    "this question. Please try rephrasing your question or upload additional "
    "documents that may contain the answer."
)


AMBIGUOUS_STANDALONE_NOUNS = {
    "method": "methods",
    "model": "models",
    "result": "results",
    "results": "results",
    "approach": "approaches",
    "technique": "techniques",
    "algorithm": "algorithms",
    "process": "processes",
    "experiment": "experiments",
    "paper": "topics",
    "figure": "figures",
    "table": "tables",
    "chapter": "chapters",
    "equation": "equations",
    "section": "sections",
}


async def check_query_clarity(question: str, chat_history: list = None) -> tuple[bool, str]:
    """
    Evaluates whether the user's question is clear and specific enough for document retrieval.
    Returns: (is_clear: bool, feedback_message: str)
    """
    raw_q = (question or "").strip()
    if not raw_q:
        return False, "Please enter a specific question about your uploaded document."

    # 1. Punctuation / Length check
    cleaned_text = re.sub(r"[^\w\s]", " ", raw_q).strip()
    words = [w.lower() for w in cleaned_text.split() if w]
    if len(words) == 0 or not any(c.isalpha() for c in raw_q):
        return False, (
            "Your question is too short or unclear. Please ask a specific question "
            "about your uploaded document so I can find the right information for you."
        )

    # 2. Summary requests are clear requests for document synthesis
    if is_summary_request(raw_q):
        return True, ""

    # 3. Conversational greetings/chitchat are handled separately
    chitchat_greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good evening",
        "who are you", "what are you", "what can you do"
    }
    if cleaned_text.lower() in chitchat_greetings:
        return True, ""

    # Check if there is an active conversational context in recent history
    has_recent_context = False
    if chat_history and len(chat_history) >= 2:
        human_msgs = [
            m for m in chat_history
            if (isinstance(m, dict) and m.get("role") == "human") or getattr(m, "type", "") == "human"
        ]
        if human_msgs:
            has_recent_context = True

    # 4. Check for pure vague phrases
    vague_phrases = {
        "what", "why", "how", "who", "when", "where",
        "explain", "tell me", "details", "more", "info", "describe", "help",
        "what is it", "how does it work", "tell me about it", "explain this",
        "explain that", "what about it", "what about this", "what about that",
        "can you explain", "give me info", "tell me more", "summarize something",
        "what does it mean", "is it good", "is it bad", "what is this", "what is that"
    }
    if cleaned_text.lower() in vague_phrases:
        if not has_recent_context:
            return False, (
                "Your question is a bit too vague to search the document accurately. "
                "Could you please specify which topic, concept, or section you would like to know about? "
                "(For example: 'What is Perceptron?' or 'How does backpropagation work?'). "
                "Providing a specific question helps me retrieve the exact information."
            )

    # 5. Meaningful content terms
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    generic_words = {"thing", "things", "stuff", "topic", "topics", "item", "items", "something", "anything"}
    non_generic = [w for w in meaningful if w not in generic_words]

    if not non_generic:
        if not has_recent_context:
            return False, (
                "It looks like your question doesn't specify a concrete subject or concept. "
                "Please mention the specific term, topic, or document section you are asking about "
                "so I can search and retrieve the right answer for you."
            )

    # 6. Standalone ambiguous nouns (e.g., "what is the method?", "explain the model")
    if len(non_generic) == 1 and non_generic[0] in AMBIGUOUS_STANDALONE_NOUNS:
        if not has_recent_context:
            noun = non_generic[0]
            plural = AMBIGUOUS_STANDALONE_NOUNS.get(noun, f"{noun}s")
            return False, (
                f"The uploaded document covers various {plural}. "
                f"Could you please specify which {noun} you are interested in? "
                "Providing the specific name or concept will allow me to retrieve the exact details."
            )

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# Node: Intent_Classifier (classify_intent)
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="Intent_Classifier")
async def classify_intent(state: RAGState) -> Dict[str, Any]:
    """Detect whether to perform document retrieval, request clarification, or general conversational AI.

    Logic:
        - If a document source is selected/uploaded (source_selected or collection_names):
          Check question clarity. If vague/underspecified, route to 'clarify'.
          Otherwise, route to 'retrieval'.
        - If no document is selected/uploaded, route to 'chitchat' (conversational AI).
    """
    has_source = bool(
        state.get("source_selected")
        or state.get("collection_names")
        or state.get("collection_name")
        or state.get("document_id")
        or state.get("docs")
    )

    if has_source:
        question = state.get("question", "")
        chat_history = state.get("chat_history", [])
        is_clear, feedback = await check_query_clarity(question, chat_history)

        if not is_clear:
            logging.info("→ classify_intent: question is vague/unclear, routing to clarify_question")
            return {
                "intent": "clarify",
                "clarification_feedback": feedback,
            }

        logging.info("→ classify_intent: source present and question is clear, routing to retrieval")
        return {"intent": "retrieval"}

    logging.info("→ classify_intent: no source selected, routing to chitchat")
    return {"intent": "chitchat"}


# ══════════════════════════════════════════════════════════════════════════════
# Node: clarify_question
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="Clarify_Question")
async def clarify_question(state: RAGState) -> Dict[str, Any]:
    """Provide constructive feedback asking the user to clarify an underspecified question."""
    feedback = state.get("clarification_feedback") or (
        "Your question seems a bit vague or underspecified. Could you please clarify "
        "what specific topic, concept, or section of the document you would like to know about? "
        "Providing more detail will help retrieve the exact information from your uploaded material."
    )
    logging.info("→ clarify_question: providing feedback to user: %s", feedback[:60])
    return {
        "raw_answer": feedback,
        "final_answer": feedback,
        "citations": "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ReAct Agent: Forced Tool Calling & Traced Execution
# ══════════════════════════════════════════════════════════════════════════════

# 1. Strict System Prompt to force tool usage
system_instruction = (
    "You are DocuVortex, an advanced AI assistant equipped with real-time tools. "
    "TOOL USAGE RULES: "
    "1. You MUST call `get_stock_price` whenever the user asks for stock quotes, share prices, or market data. "
    "2. You MUST call `calculator` for any mathematical operations or arithmetic calculations. "
    "3. You MUST call `search_tool` for current events, latest news, recent developments, real-world facts, or whenever the user asks to look up, search, or check something on the web. Do NOT rely on static memory for current or verifiable facts. "
    "4. For weather inquiries, you MUST use the `get_weather` tool. "
    "5. For requests regarding research papers, academic studies, or scientific literature, you MUST use the `search_arxiv` tool. "
    "NEVER emit conversational preambles (e.g., 'Let me check the weather...'). Call all tools silently. "
    "When tools return results, synthesize the findings into a clear, concise, and helpful response."
)


# 2. Initialize the prebuilt ReAct agent
def _build_react_generation_llm(model_override: str = None):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        from langchain_groq import ChatGroq
        model_name = model_override or os.getenv("GROQ_AGENT_MODEL", "qwen/qwen3.8-27b")
        return ChatGroq(
            model=model_name,
            temperature=0.2,
            api_key=groq_api_key,
            groq_api_key=groq_api_key,
            max_tokens=1024,
        )

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            temperature=0.2,
            google_api_key=google_api_key,
            max_output_tokens=1024,
        )

    raise ValueError("Neither GROQ_API_KEY nor GOOGLE_API_KEY is configured.")


def _create_agent_graph(llm, tools, prompt_str):
    """Instantiate prebuilt ReAct agent supporting both modern and legacy parameter names."""
    try:
        return create_react_agent(llm, tools=tools, prompt=prompt_str)
    except TypeError:
        return create_react_agent(llm, tools=tools, state_modifier=prompt_str)


generation_llm = _build_react_generation_llm()
react_agent_graph = _create_agent_graph(generation_llm, chitchat_tools, system_instruction)


@traceable(name="ReAct_Chitchat_Agent")
async def run_react_agent(state: RAGState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute prebuilt LangGraph ReAct agent with config and LangSmith naming."""
    query_text = state.get("query") or state.get("question") or ""
    chat_history = state.get("chat_history", [])

    messages = list(chat_history) if chat_history else []
    messages.append(HumanMessage(content=query_text))

    inputs = {"messages": messages}

    # Pass the config down to ensure streaming and tracing work
    if isinstance(config, dict):
        config["run_name"] = "ReAct_Chitchat_Agent"

    try:
        response = await react_agent_graph.ainvoke(inputs, config)
        messages = response.get("messages", [])
        final_answer = ""
        if messages:
            # Find the last message with actual content
            for msg in reversed(messages):
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    content = extract_message_text(content)
                text = str(content).strip()
                if text:
                    final_answer = text
                    break

        if not final_answer:
            final_answer = "I couldn't generate an answer. Please try asking in a different way."

        return {
            "generation": final_answer,
            "raw_answer": final_answer,
            "final_answer": final_answer,
            "citations": "",
        }
    except Exception as e:
        logging.exception("ReAct agent execution failed: %s", e)
        # Resilient fallback with secondary model
        try:
            fallback_llm = _build_react_generation_llm(model_override="openai/gpt-oss-120b")
            fallback_agent = _create_agent_graph(fallback_llm, chitchat_tools, system_instruction)
            fb_res = await fallback_agent.ainvoke(inputs, config)
            fb_messages = fb_res.get("messages", [])
            fb_ans = ""
            for msg in reversed(fb_messages):
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    content = extract_message_text(content)
                text = str(content).strip()
                if text:
                    fb_ans = text
                    break

            if fb_ans:
                return {
                    "generation": fb_ans,
                    "raw_answer": fb_ans,
                    "final_answer": fb_ans,
                    "citations": "",
                }
        except Exception as fb_err:
            logging.error("Fallback agent also failed: %s", fb_err)

        err_msg = "I encountered an issue processing your request. Please try again."
        return {
            "generation": err_msg,
            "raw_answer": err_msg,
            "final_answer": err_msg,
            "citations": "",
        }


# Aliases for graph builder and external callers
chitchat = run_react_agent
run_chitchat = run_react_agent


# ══════════════════════════════════════════════════════════════════════════════
# Node: Document_Grader (grade_documents)
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="Document_Grader")
async def grade_documents(state: RAGState) -> Dict[str, Any]:
    """Evaluate whether retrieved chunks contain relevant information.

    Strategy to avoid Gemini 429 errors and context size issues:
        - Pass condensed summaries (first 200 chars per chunk + keywords)
        - Cap at 5 chunks maximum
        - max_output_tokens=10 (just 'yes' or 'no')
        - On ANY failure, gracefully degrade: skip grading, assume relevant
    """
    docs = state.get("docs", [])
    question = state["question"]

    if not docs:
        logging.info("→ grade_documents: no docs retrieved, marking irrelevant")
        return {"grade": "irrelevant"}

    try:
        # Condense: first 200 chars per chunk, max 5 chunks
        condensed = "\n".join(
            f"[Chunk {i+1}]: {doc.page_content[:200]}..."
            for i, doc in enumerate(docs[:5])
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            max_output_tokens=10,
        )

        result = await llm.ainvoke(
            "You are a relevance grader. Given a user question and document context, "
            "determine if the context contains information relevant to answering the question.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{condensed}\n\n"
            "Reply with ONLY one word: 'yes' if relevant, 'no' if irrelevant."
        )

        answer = result.content.strip().lower()
        grade = "relevant" if "yes" in answer else "irrelevant"
        logging.info("→ grade_documents: %s (raw=%r, %d chunks evaluated)", grade, answer, min(len(docs), 5))
        return {"grade": grade}

    except Exception as e:
        # Graceful degradation: skip grading on any error (429, context too large, etc.)
        logging.warning(
            "→ grade_documents: grading failed (%s), skipping — routing to generate", e
        )
        return {"grade": "relevant"}


# ══════════════════════════════════════════════════════════════════════════════
# Node: fallback_response
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="fallback_response")
async def fallback_response(state: RAGState) -> Dict[str, Any]:
    """Return a polite message when documents don't contain relevant information."""
    logging.info("→ fallback_response: no relevant docs found")
    return {
        "raw_answer": FALLBACK_ANSWER,
        "final_answer": FALLBACK_ANSWER,
        "citations": "",
    }
