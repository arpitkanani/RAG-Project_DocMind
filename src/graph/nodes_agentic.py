"""Consolidated graph nodes for the DocuVortex agentic RAG workflow.

Nodes:
    classify_intent   – Detect conversational vs retrieval intent
    chitchat          – Lightweight conversational response (Gemini 3.6 Flash)
    grade_documents   – Evaluate retrieved chunk relevance (Gemini 3.6 Flash)
    fallback_response – Polite message when docs are irrelevant
"""

import json
import os
import re
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langgraph.prebuilt import create_react_agent
from langsmith import traceable  # type: ignore

from src.chains.qa_chain import FALLBACK_ANSWER, is_summary_request
from src.components.retriever import STOP_WORDS
from src.exception import CustomException
from src.graph.helpers import extract_message_text
from src.graph.state import RAGState
from src.graph.tools import chitchat_tools, get_stock_price
from src.logger import logging


# ─── Gemini 3.8 Flash / ReAct Model Setup ───
_gemini_flash = None


def _get_gemini_flash():
    """Lazy-initialize the Flash model with multi-model fallbacks (Groq primary, Gemini secondary)."""
    global _gemini_flash
    if _gemini_flash is None:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")

        instances = []

        if groq_api_key:
            from langchain_groq import ChatGroq
            for m in [os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-20b"), "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]:
                try:
                    instances.append(
                        ChatGroq(
                            model=m,
                            temperature=0,
                            api_key=groq_api_key,
                            groq_api_key=groq_api_key,
                            max_tokens=1024,
                        )
                    )
                except Exception:
                    pass

        if google_api_key and not google_api_key.startswith("AQ."):
            for m in ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"]:
                try:
                    instances.append(
                        ChatGoogleGenerativeAI(
                            model=m,
                            temperature=0,
                            google_api_key=google_api_key,
                            max_output_tokens=1024,
                        )
                    )
                except Exception:
                    pass

        if not instances:
            raise ValueError("Neither GOOGLE_API_KEY nor GROQ_API_KEY is configured.")

        primary = instances[0]
        _gemini_flash = primary.with_fallbacks(instances[1:]) if len(instances) > 1 else primary
    return _gemini_flash


FALLBACK_MESSAGE = (
    "I am only able to provide answers directly grounded in your uploaded document(s). "
    "I couldn't find information about that in the uploaded material."
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

    # 2b. Spelling tolerance: queries with 4+ words and at least 1 meaningful term
    # are ALWAYS clear enough — spelling mistakes should never trigger clarification
    meaningful_early = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    if len(words) >= 4 and len(meaningful_early) >= 1:
        return True, ""

    # 2c. If 2+ meaningful non-generic words, the query has enough specificity
    generic_words_early = {"thing", "things", "stuff", "topic", "topics", "item", "items", "something", "anything"}
    non_generic_early = [w for w in meaningful_early if w not in generic_words_early]
    if len(non_generic_early) >= 2:
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

# Patterns indicating an explicit tool inquiry or conversational greeting
TOOL_OR_CHITCHAT_PATTERNS = [
    # Stock / financial quotes & tickers (global & Indian stocks)
    r"\b(stocks?|shares?|tickers?|stock\s*prices?|share\s*prices?|stock\s*quotes?|ticker\s*symbol|market\s*cap|market\s*quote|nasdaq|dow\s*jones|s&p\s*500|aapl|tsla|msft|nvda|googl|amzn|meta|dividend|nifty|sensex|bse|nse|mahindra|reliance|tcs|infosys|hdfc|icici|axis\s*bank|sbi|tata|bitcoin|crypto)\b",
    # Weather & climate (including common typos like 'wheater' and 'whether in ...')
    r"\b(weather|wheater|temperature|temp\s*in|forecast|humidity|climate|rain(ing)?(\s*today)?|whether\s+(in|of|for|at|today|tomorrow|like))\b",
    # Math & Calculations
    r"\b(calculate|computation|arithmetic)\b",
    r"^\s*[\d\.\(\)\+\-\*\/\^\s]+\s*$",
    r"\b\d+\s*[\+\-\*\/]\s*\d+\b",
    # ArXiv & scientific papers
    r"\b(search\s*arxiv|find\s*papers\s*on|research\s*papers?\s*on|arxiv)\b",
    # Live web search & current news
    r"\b(search\s*(the\s*)?web|latest\s*news|current\s*events|news\s*today|google\s*search)\b",
    # Conversational greetings & assistant identity
    r"^\s*(hi|hello|hey|greetings|good\s*morning|good\s*afternoon|good\s*evening|who\s*are\s*you|what\s*are\s*you|what\s*can\s*you\s*do|how\s*are\s*you|help)\s*[\?\!\.]*\s*$",
]


def _was_last_response_clarification(chat_history: list) -> bool:
    """Check if the last AI response was a clarification prompt."""
    if not chat_history:
        return False
    for msg in reversed(chat_history):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in ("ai", "assistant"):
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            try:
                data = json.loads(str(content))
                if isinstance(data, dict) and data.get("type") == "clarification":
                    return True
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            return False  # Last AI message was a normal response
    return False


@traceable(name="Intent_Classifier")
async def classify_intent(state: RAGState) -> Dict[str, Any]:
    """Detect whether to perform document retrieval, request clarification, or general conversational AI.

    Logic:
        - If query matches explicit real-time tools (stock, weather, math, search, greeting) -> 'chitchat'.
        - If a document source is selected/uploaded:
          Check question clarity. If vague/underspecified, route to 'clarify'.
          Otherwise, route to 'retrieval'.
        - If no document is selected/uploaded, route to 'chitchat' (conversational AI).
        - Never clarify twice in a row (anti-loop guard).
    """
    question = state.get("question") or state.get("query") or ""
    q_lower = question.lower().strip()

    # 1. Real-time tools & conversational queries always take priority
    for pattern in TOOL_OR_CHITCHAT_PATTERNS:
        if re.search(pattern, q_lower):
            logging.info("→ classify_intent: matched tool/chitchat pattern '%s', routing to chitchat", pattern)
            return {"intent": "chitchat"}

    # 2. Check if a document source is active in this session
    has_source = bool(
        state.get("source_selected")
        or state.get("collection_names")
        or state.get("collection_name")
        or state.get("document_id")
        or state.get("docs")
    )

    if has_source:
        chat_history = state.get("chat_history", [])

        # GUARD: Never clarify twice in a row — if the last AI response was
        # a clarification, proceed directly to retrieval regardless of query clarity
        if _was_last_response_clarification(chat_history):
            logging.info("→ classify_intent: skipping clarification (last response was clarify)")
            return {"intent": "retrieval"}

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
# Node: clarify_question — Structured Clarification with Options
# ══════════════════════════════════════════════════════════════════════════════


async def _generate_clarification_options(question: str) -> list:
    """Generate 2-4 refined query suggestions for a vague question.

    Uses a lightweight LLM call to generate contextual suggestions.
    Falls back to template-based suggestions on failure.
    """
    try:
        llm = _get_gemini_flash()
        prompt = (
            f'The user asked a vague question: "{question}"\n'
            "Generate exactly 3 specific, refined versions of this question "
            "that would work well for document search.\n"
            "Return ONLY a JSON array of strings, nothing else.\n"
            'Example: ["What is the perceptron learning algorithm?", '
            '"How does backpropagation work?", '
            '"What are the key experimental results?"]'
        )
        from src.utils.helpers import extract_text

        result = await llm.ainvoke(prompt)
        content = extract_text(getattr(result, "content", result))
        # Extract JSON array from response (handle markdown fences)
        if "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()
        options = json.loads(content)
        if isinstance(options, list) and len(options) >= 2:
            return options[:4]
    except Exception as e:
        logging.debug("Clarification option generation fallback: %s", e)

    # Template-based fallback
    q = question.strip()
    return [
        f"What is {q}?",
        f"Explain {q} in detail",
        f"Summarize the key points about {q}",
    ]


@traceable(name="Clarify_Question")
async def clarify_question(state: RAGState) -> Dict[str, Any]:
    """Provide interactive clarification with clickable options."""
    question = state.get("question", "")
    feedback = state.get("clarification_feedback") or (
        "Your question is a bit vague. Could you be more specific? "
        "Here are some suggestions:"
    )

    options = await _generate_clarification_options(question)

    clarify_data = {
        "type": "clarification",
        "message": feedback,
        "options": options,
    }

    clarify_json = json.dumps(clarify_data)
    logging.info("→ clarify_question: providing %d options for: %s", len(options), question[:50])
    return {
        "raw_answer": clarify_json,
        "final_answer": clarify_json,
        "citations": "",
        "intent": "clarify",
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


# 2. ReAct Chitchat Agent powered by chitchat_subgraph
@traceable(name="ReAct_Chitchat_Agent")
async def run_react_agent(state: RAGState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Execute prebuilt tool-calling chitchat subgraph with config and LangSmith naming."""
    from src.graph.chitchat_subgraph import chitchat_subgraph

    query_text = state.get("query") or state.get("question") or ""
    chat_history = state.get("chat_history", [])

    # Filter out system messages so document prompts do not interfere with the agent tools
    messages = []
    if chat_history:
        for m in chat_history:
            if isinstance(m, (HumanMessage, AIMessage)):
                messages.append(m)
            elif isinstance(m, dict):
                role = m.get("role") or m.get("type")
                content = m.get("content") or ""
                if role in ("human", "user"):
                    messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query_text))

    if isinstance(config, dict):
        config["run_name"] = "ReAct_Chitchat_Agent"

    try:
        response = await chitchat_subgraph.ainvoke(
            {"messages": messages, "iteration_count": 0, "tools_called": False},
            config,
        )
        sub_messages = response.get("messages", [])
        final_answer = ""
        if sub_messages:
            for msg in reversed(sub_messages):
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    content = extract_message_text(content)
                text = str(content).strip()
                if text and not getattr(msg, "tool_calls", None):
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
        logging.exception("ReAct chitchat subgraph execution failed: %s", e)

        # Resilient fallback 1: Direct stock tool execution if query requested stock or market quote
        q_lower = query_text.lower()
        if any(w in q_lower for w in ["stock", "share", "ticker", "quote", "price"]):
            try:
                words = [w for w in re.findall(r"\b[A-Za-z0-9]+\b", query_text) if w.lower() not in {"what", "is", "the", "stock", "price", "of", "share", "today", "for", "current", "check"}]
                sym = words[-1] if words else "AAPL"
                res = get_stock_price.invoke({"symbol": sym})
                if isinstance(res, dict):
                    if res.get("current_price") and res.get("current_price") not in ("Quote Unavailable", "Market Quote (Live)"):
                        ans = f"The latest stock price for **{res.get('symbol', sym)}** is **{res.get('current_price')}** (Change: {res.get('change', 'N/A')}, Day High: {res.get('day_high', 'N/A')}, Day Low: {res.get('day_low', 'N/A')}, Exchange: {res.get('exchange', 'US')})."
                    elif res.get("live_search_quotes"):
                        ans = f"Here is the latest market information for **{res.get('symbol', sym)}**:\n\n{res['live_search_quotes']}"
                    elif res.get("message"):
                        ans = res["message"]
                    else:
                        ans = f"Could not retrieve stock data for {sym} at this moment."
                    return {
                        "generation": ans,
                        "raw_answer": ans,
                        "final_answer": ans,
                        "citations": "",
                    }
            except Exception as direct_err:
                logging.error("Direct stock tool fallback failed: %s", direct_err)

        # Resilient fallback 1.5: Direct weather tool execution if query requested weather
        if any(w in q_lower for w in ["weather", "wheater", "temperature", "forecast", "climate", "humidity", "rain"]):
            try:
                from src.graph.tools import get_weather
                words = [w for w in re.findall(r"\b[A-Za-z0-9]+\b", query_text) if w.lower() not in {"what", "is", "the", "weather", "wheater", "temperature", "forecast", "in", "of", "for", "at", "today", "now", "current", "how", "like"}]
                loc = " ".join(words) if words else "London"
                weather_res = await get_weather.ainvoke({"location": loc})
                if weather_res and isinstance(weather_res, str):
                    return {
                        "generation": weather_res,
                        "raw_answer": weather_res,
                        "final_answer": weather_res,
                        "citations": "",
                    }
            except Exception as direct_w_err:
                logging.error("Direct weather tool fallback failed: %s", direct_w_err)

        # Resilient fallback 2: Direct web search tool execution
        try:
            from src.graph.tools import search_tool
            res = search_tool.invoke({"query": query_text})
            if res and isinstance(res, str) and len(res.strip()) > 10:
                return {
                    "generation": res,
                    "raw_answer": res,
                    "final_answer": res,
                    "citations": "",
                }
        except Exception as search_err:
            logging.error("Direct web search fallback failed: %s", search_err)

        err_msg = "I encountered an issue processing your request. Please try asking again."
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

        llm = _get_gemini_flash()

        result = await llm.ainvoke(
            "You are a relevance grader. Given a user question and document context, "
            "determine if the context contains information relevant to answering the question.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{condensed}\n\n"
            "Reply with ONLY one word: 'yes' if relevant, 'no' if irrelevant."
        )

        answer = extract_text(result.content if hasattr(result, "content") else result).strip().lower()
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
