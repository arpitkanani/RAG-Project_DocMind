from datetime import datetime
import os
import re
import secrets
import sys
from typing import Annotated, Any, Dict, Sequence, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.exception import CustomException
from src.graph.chitchat_tools import chitchat_tools
from src.logger import logging
from src.utils.helpers import extract_text
from src.utils.rate_limiter import llm_rate_limiter


class ChitChatState(TypedDict):
    """Isolated state for the chitchat tool-calling subgraph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration_count: int
    tools_called: bool


TEMPORAL_PATTERNS = re.compile(
    r"\b(latest|current|recent|most recent|updated|who won|winner|champions?|trophy|cup|tournament|status|this year|last year|next year|today|yesterday|tomorrow|upcoming|202[0-9]|203[0-9])\b",
    re.IGNORECASE,
)


def _get_current_date_str() -> str:
    """Dynamically format the current absolute system date at runtime."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")


def _get_dynamic_chitchat_prompt() -> str:
    """Base system prompt dynamically injected with current runtime date and strict tool rules."""
    current_date = _get_current_date_str()
    return (
        "You are DocuVortex, an intelligent, helpful, and friendly AI assistant.\n\n"
        f"Current Date: {current_date}\n\n"
        "STRICT TOOL SELECTION & FRESHNESS RULES:\n"
        "1. Stock Prices / Quotes / Crypto: When asked about the stock price, share price, quote, or ticker of ANY company or asset (e.g., Tesla, Apple, Mahindra, Axis Bank, Tata, Reliance, Bitcoin, Ethereum): "
        "YOU MUST CALL 'get_stock_price'. NEVER call 'search_tool' for stock prices.\n"
        "2. Weather / Temperature / Climate: When asked about the weather, temperature, humidity, rain, or climate for any city or location (e.g., Mumbai, New York, London, Delhi, Paris): "
        "YOU MUST CALL 'get_weather'. NEVER call 'search_tool' for weather.\n"
        "3. Math / Calculations: When asked to calculate or evaluate arithmetic (e.g., 230*460): "
        "YOU MUST CALL 'calculator'. NEVER call 'search_tool' for math.\n"
        "4. Academic Papers: When asked for scientific preprints or research papers, call 'search_arxiv'.\n"
        "5. TIME-SENSITIVE / FRESHNESS / HISTORICAL TIMELINE & CURRENT EVENTS (CRITICAL):\n"
        "   - Your internal weights and parametric memory are COMPLETELY UNRELIABLE and frozen for current facts, recent tournament results, winners, sports championships, and chronological events.\n"
        "   - YOU MUST CALL 'search_tool' whenever the query contains temporal or freshness markers such as 'latest', 'recent', 'most recent', 'current', 'updated', 'who won', 'winner', 'status', 'this year', 'last year', or explicit years (e.g., 2024, 2025, 2026).\n"
        "   - NEVER answer from internal memory or declare an event unanswerable without executing 'search_tool' first.\n\n"
        "OPTIMIZED SEARCH QUERY GENERATION:\n"
        "When calling 'search_tool' for temporal questions, formulate an objective, factual search string. "
        "For example, if the user asks 'who won the latest Champion trophy', generate query: 'most recent ICC Champions Trophy winner results history'.\n\n"
        "CRITICAL CONVERSATIONAL RULE:\n"
        "Answer ONLY the user's latest question directly. DO NOT mention, repeat, or summarize past questions, previous tool results, or earlier conversation topics. Keep your answers focused, direct, clean, and helpful."
    )


def _get_dynamic_response_structure_prompt() -> str:
    """Synthesis prompt with dynamic current date and search-payload anti-hallucination priority."""
    current_date = _get_current_date_str()
    return (
        "You are DocuVortex, an intelligent, helpful AI assistant.\n"
        f"Current Date: {current_date}\n\n"
        "Your task is to provide a clean, direct, and well-structured answer to ONLY the user's latest question using the tool results.\n\n"
        "CRITICAL ANTI-HALLUCINATION & FRESHNESS RULES:\n"
        "1. SEARCH TOOL DATA OVERRULES INTERNAL MEMORY: Your internal training memory is frozen and lacks recent tournament and current event results. You MUST strictly prioritize and verify facts against the live search tool results.\n"
        "2. If the user asks about the 'latest' or 'most recent' tournament/event, check the search results for the most recent edition held. Never fall back to older tournaments (like 2017) if newer tournaments (like 2025) are mentioned in search or if a future tournament has not yet occurred.\n"
        "3. If a tournament (such as 2026 T20 World Cup) has not yet taken place relative to the Current Date, state clearly that it is scheduled for that year and has not yet been played.\n"
        "4. Answer ONLY what the user asked in their current question. Do NOT discuss, summarize, or bring up past conversation topics or prior queries.\n"
        "5. Present the information clearly and attractively using bullet points, bold key figures, and clean markdown.\n"
        "6. Do NOT mention internal tool names (like get_weather, get_stock_price, search_tool) or raw JSON."
    )


def _optimize_search_query(user_query: str) -> str:
    """Reformulate conversational questions with temporal markers into objective, factual search queries."""
    q = (user_query or "").strip()
    q_clean = re.sub(
        r"(?i)^(can you\s+)?(please\s+)?(tell\s+me\s+)?(who\s+won|what\s+is|what's|which\s+team\s+won|who\s+is\s+the\s+winner\s+of)\s+",
        "",
        q,
    )
    q_clean = re.sub(r"[?!.]+$", "", q_clean).strip()

    lowered = q_clean.lower()
    if "champion" in lowered and "trophy" in lowered and "icc" not in lowered:
        q_clean = f"ICC {q_clean}"
    if "t20" in lowered and "world" in lowered and "icc" not in lowered:
        q_clean = f"ICC Men's {q_clean}"

    if not any(kw in lowered for kw in ["winner", "results", "history", "final", "schedule"]):
        q_clean = f"{q_clean} winner results"

    return q_clean.strip()


def _should_force_search(query: str) -> bool:
    """Check if query is time-sensitive and should automatically force DuckDuckGo search."""
    q = (query or "").lower().strip()
    if not q:
        return False
    # If weather inquiry, keep weather tool as primary
    if any(w in q for w in ["weather", "temperature", "forecast", "climate", "rain", "humidity"]):
        return False
    # If stock inquiry, keep stock tool as primary
    if any(w in q for w in ["stock", "share price", "ticker", "nasdaq", "nifty", "sensex", "crypto", "bitcoin", "ethereum"]):
        return False
    # If pure math expression, keep calculator as primary
    if re.search(r"^\s*[\d\.\s\+\-\*\/\(\)]+\s*$", q) or any(op in q for op in ["calculate", "multiply", "divide", "add", "subtract"]):
        return False
    return bool(TEMPORAL_PATTERNS.search(q))


_GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3.8-27b",
]

_GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]


def _build_chitchat_tool_llm():
    """Build primary tool-calling LLM with resilient multi-model and cross-provider fallbacks."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    groq_model_pref = os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-20b")
    ordered_groq = [groq_model_pref] + [m for m in _GROQ_FALLBACK_MODELS if m != groq_model_pref]

    candidates = []
    # Groq models
    if groq_api_key:
        for m in ordered_groq:
            candidates.append(
                ChatGroq(model=m, temperature=0.2, groq_api_key=groq_api_key, max_tokens=1024)
            )
    # Gemini models
    if google_api_key and not google_api_key.startswith("AQ."):
        for m in _GEMINI_FALLBACK_MODELS:
            candidates.append(
                ChatGoogleGenerativeAI(model=m, temperature=0.2, google_api_key=google_api_key, max_output_tokens=1024)
            )

    if not candidates:
        raise ValueError("Neither GROQ_API_KEY nor GOOGLE_API_KEY is configured.")

    primary = candidates[0].bind_tools(chitchat_tools)
    if len(candidates) > 1:
        fallbacks = [c.bind_tools(chitchat_tools) for c in candidates[1:]]
        return primary.with_fallbacks(fallbacks)
    return primary


def _build_chitchat_synthesis_llm():
    """Build synthesis LLM with multi-model fallbacks for structured answer formatting."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    groq_model_pref = os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-20b")
    ordered_groq = [groq_model_pref] + [m for m in _GROQ_FALLBACK_MODELS if m != groq_model_pref]

    candidates = []
    if groq_api_key:
        for m in ordered_groq:
            candidates.append(
                ChatGroq(model=m, temperature=0.2, groq_api_key=groq_api_key, max_tokens=1024)
            )
    if google_api_key and not google_api_key.startswith("AQ."):
        for m in _GEMINI_FALLBACK_MODELS:
            candidates.append(
                ChatGoogleGenerativeAI(model=m, temperature=0.2, google_api_key=google_api_key, max_output_tokens=1024)
            )

    if not candidates:
        raise ValueError("Neither GROQ_API_KEY nor GOOGLE_API_KEY is configured.")

    primary = candidates[0]
    if len(candidates) > 1:
        return primary.with_fallbacks(candidates[1:])
    return primary


async def chitchat_agent_node(state: ChitChatState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Agent node: decides whether to call tools or generate response."""
    try:
        raw_messages = list(state.get("messages", []))
        # Extract latest human query for temporal and tool verification
        latest_query = ""
        for m in reversed(raw_messages):
            if isinstance(m, HumanMessage):
                latest_query = extract_text(m.content)
                break

        # Ensure dynamic SystemMessage (with runtime date) is strictly at index 0 (Groq/OpenAI requirement)
        non_system = [m for m in raw_messages if not isinstance(m, SystemMessage)]
        system_content = _get_dynamic_chitchat_prompt()
        messages = [SystemMessage(content=system_content)] + non_system

        iteration = state.get("iteration_count", 0) + 1
        tools_called = state.get("tools_called", False)

        llm_with_tools = _build_chitchat_tool_llm()
        await llm_rate_limiter.aacquire()
        response = await llm_with_tools.ainvoke(messages, config)

        tool_calls = getattr(response, "tool_calls", None)

        # Programmatic guardrail: if time-sensitive query did not trigger a tool call on iteration 1, force search_tool
        if iteration == 1 and not tool_calls and _should_force_search(latest_query):
            search_q = _optimize_search_query(latest_query)
            logging.info(
                "chitchat_agent: programmatically forcing search_tool for temporal query: %r -> %r",
                latest_query,
                search_q,
            )
            call_id = f"call_{secrets.token_hex(8)}"
            forced_call = {
                "name": "search_tool",
                "args": {"query": search_q},
                "id": call_id,
                "type": "tool_call",
            }
            response = AIMessage(
                content="",
                tool_calls=[forced_call],
            )
            tool_calls = [forced_call]

        if tool_calls:
            tools_called = True
            logging.info("chitchat_agent requested tool calls: %s (iteration %d/5)", 
                         [tc.get("name") for tc in tool_calls], iteration)
        else:
            logging.info("chitchat_agent generated direct response (iteration %d)", iteration)

        return {
            "messages": [response],
            "iteration_count": iteration,
            "tools_called": tools_called,
        }
    except Exception as e:
        logging.exception("chitchat_agent_node encountered an error: %s", e)
        raise CustomException(e, sys)


async def structure_answer_node(state: ChitChatState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Pass latest turn tool outputs and user query to LLM to create a polished, focused answer."""
    messages = list(state.get("messages", []))
    try:
        # Isolate the current turn: find the last HumanMessage so past turns are NEVER summarized
        last_human_idx = -1
        for i, m in enumerate(messages):
            if isinstance(m, HumanMessage):
                last_human_idx = i

        if last_human_idx != -1:
            current_turn_messages = [m for m in messages[last_human_idx:] if not isinstance(m, SystemMessage)]
            latest_query = messages[last_human_idx].content
        else:
            current_turn_messages = [m for m in messages if not isinstance(m, SystemMessage)]
            latest_query = ""

        # System prompt MUST be at index 0 for Groq/OpenAI APIs (with dynamic date & search-overrule rules)
        synthesis_system = SystemMessage(content=_get_dynamic_response_structure_prompt())
        prompt_message = HumanMessage(
            content=f"Please synthesize the tool results into a clean, direct, and well-structured answer to ONLY this question: '{latest_query}'. Overwrite internal training weights with live search results where applicable. Do NOT include, mention, or summarize any past conversation topics or prior queries."
        )
        synthesis_messages = [synthesis_system] + current_turn_messages + [prompt_message]

        llm = _build_chitchat_synthesis_llm()
        await llm_rate_limiter.aacquire()
        final_response = await llm.ainvoke(synthesis_messages, config)
        clean_text = extract_text(final_response.content if hasattr(final_response, "content") else final_response)
        final_response.content = clean_text

        logging.info("structure_answer_node: generated structured response (%d chars)", 
                     len(clean_text))
        return {"messages": [final_response]}
    except Exception as e:
        logging.exception("structure_answer_node encountered error: %s", e)
        # Resilient fallback: return the last non-empty assistant message in current turn if synthesis fails
        target_messages = current_turn_messages if 'current_turn_messages' in locals() and current_turn_messages else messages
        for m in reversed(target_messages):
            if getattr(m, "content", None) and not getattr(m, "tool_calls", None):
                m.content = extract_text(m.content)
                return {"messages": [m]}
        raise CustomException(e, sys)


def should_continue_chitchat(state: ChitChatState) -> str:
    """Enforce tool calling loop up to 5 iterations, then route to structure_answer or end."""
    iteration = state.get("iteration_count", 0)
    messages = state.get("messages", [])
    tools_called = state.get("tools_called", False)

    if not messages:
        return END

    last_message = messages[-1]
    has_tool_calls = bool(getattr(last_message, "tool_calls", None))

    # If the LLM requested tools and we haven't reached the 5-iteration limit, execute tools
    if has_tool_calls and iteration < 5:
        return "tools"

    # If tools were executed in this subgraph run, route to structure_answer for final synthesis
    if tools_called or has_tool_calls:
        return "structure_answer"

    # Pure conversational turn with no tools: agent output is already final
    return END


# ToolNode handles tool errors gracefully without crashing the graph
chitchat_tool_node = ToolNode(tools=chitchat_tools, handle_tool_errors=True)


def build_chitchat_subgraph():
    """Compile self-contained subgraph for chitchat with tool execution and answer structuring."""
    builder = StateGraph(ChitChatState)

    builder.add_node("chitchat_agent", chitchat_agent_node)
    builder.add_node("tools", chitchat_tool_node)
    builder.add_node("structure_answer", structure_answer_node)

    builder.add_edge(START, "chitchat_agent")
    builder.add_conditional_edges(
        "chitchat_agent",
        should_continue_chitchat,
        {
            "tools": "tools",
            "structure_answer": "structure_answer",
            END: END,
        },
    )
    # After tool execution, loop back to agent (up to 5 iterations max)
    builder.add_edge("tools", "chitchat_agent")
    # After structuring final answer, finish subgraph execution
    builder.add_edge("structure_answer", END)

    return builder.compile()


chitchat_subgraph = build_chitchat_subgraph()
