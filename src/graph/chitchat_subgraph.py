import os
import sys
from typing import Annotated, Any, Dict, Sequence, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.exception import CustomException
from src.graph.chitchat_tools import chitchat_tools
from src.logger import logging
from src.utils.rate_limiter import llm_rate_limiter


class ChitChatState(TypedDict):
    """Isolated state for the chitchat tool-calling subgraph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration_count: int
    tools_called: bool


CHITCHAT_SYSTEM_PROMPT = (
    "You are DocuVortex, an intelligent, helpful, and friendly AI assistant. "
    "No document has been selected for this query. "
    "You have access to tools:\n"
    "- 'duckduckgo_search' for searching recent news, facts, and live web information.\n"
    "- 'calculator' for doing math computations (operations: add, sub, mul, div).\n"
    "- 'get_stock_price' for retrieving the latest stock market prices and quotes for ticker symbols (e.g., AAPL, TSLA, MSFT).\n\n"
    "Use these tools whenever relevant to answer the user's questions accurately. "
    "If no tool is required, answer conversationally and helpfully from your knowledge without mentioning documents."
)

RESPONSE_STRUCTURE_PROMPT = (
    "You are DocuVortex, an intelligent AI assistant. "
    "Review the tool results and user question in the conversation. "
    "Synthesize the tool findings into a clean, well-structured, clear, and comprehensive answer for the user. "
    "Format with headings, bullet points, or bold text where appropriate. Do not output raw JSON or internal tool details."
)


def _build_agent_groq_llm(model_override: str = None) -> ChatGroq:
    """Build dedicated Groq LLM with high precision for tool calling."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    model_name = model_override or os.getenv("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=model_name,
        temperature=0.2,
        groq_api_key=groq_api_key,
        max_tokens=1024,
    )


async def chitchat_agent_node(state: ChitChatState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Agent node: decides whether to call tools or generate response."""
    try:
        raw_messages = list(state.get("messages", []))
        # Ensure SystemMessage is strictly at index 0 (Groq/OpenAI requirement)
        non_system = [m for m in raw_messages if not isinstance(m, SystemMessage)]
        messages = [SystemMessage(content=CHITCHAT_SYSTEM_PROMPT)] + non_system

        iteration = state.get("iteration_count", 0) + 1
        tools_called = state.get("tools_called", False)

        try:
            llm = _build_agent_groq_llm()
            llm_with_tools = llm.bind_tools(chitchat_tools)
            await llm_rate_limiter.aacquire()
            response = await llm_with_tools.ainvoke(messages, config)
        except Exception as groq_err:
            logging.warning("chitchat_agent primary model failed: %s. Trying llama-3.1-8b-instant.", groq_err)
            fallback_llm = _build_agent_groq_llm(model_override="llama-3.1-8b-instant")
            llm_with_tools = fallback_llm.bind_tools(chitchat_tools)
            await llm_rate_limiter.aacquire()
            response = await llm_with_tools.ainvoke(messages, config)

        tool_calls = getattr(response, "tool_calls", None)
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
    """Pass conversation history and tool outputs to LLM to create a polished, structured answer."""
    messages = list(state.get("messages", []))
    try:
        # System prompt MUST be at index 0 for Groq/OpenAI APIs
        synthesis_system = SystemMessage(content=RESPONSE_STRUCTURE_PROMPT)
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        prompt_message = HumanMessage(
            content="Based on the above tool results and conversation, please synthesize the findings into a clear, comprehensive, and well-structured answer."
        )
        synthesis_messages = [synthesis_system] + non_system_messages + [prompt_message]

        try:
            llm = _build_agent_groq_llm()
            await llm_rate_limiter.aacquire()
            final_response = await llm.ainvoke(synthesis_messages, config)
        except Exception as groq_err:
            logging.warning("structure_answer primary model failed: %s. Falling back to llama-3.1-8b-instant.", groq_err)
            fallback_llm = _build_agent_groq_llm(model_override="llama-3.1-8b-instant")
            await llm_rate_limiter.aacquire()
            final_response = await fallback_llm.ainvoke(synthesis_messages, config)

        logging.info("structure_answer_node: generated structured response (%d chars)", 
                     len(str(final_response.content)))
        return {"messages": [final_response]}
    except Exception as e:
        logging.exception("structure_answer_node encountered error: %s", e)
        # Resilient fallback: return the last non-empty assistant message if synthesis fails
        for m in reversed(messages):
            if getattr(m, "content", None) and not getattr(m, "tool_calls", None):
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
