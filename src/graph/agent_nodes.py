import sys
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage  # type:ignore
from langgraph.graph import END  # type:ignore
from langgraph.prebuilt import ToolNode  # type:ignore

from src.chains.qa_chain import _build_llm, is_summary_request
from src.exception import CustomException
from src.graph.helpers import extract_message_text
from src.graph.state import AgentState
from src.graph.tools import AVAILABLE_TOOLS
from src.logger import logging
from src.utils.rate_limiter import llm_rate_limiter

SYSTEM_PROMPT = """You are DocMind, a document assistant that answers ONLY from
the user's uploaded documents. You have no general knowledge.

RULES:
1. ALWAYS call rag_query for questions about document content.
2. ALWAYS call summarize_document when the user asks for a summary or overview.
3. NEVER answer from general knowledge. If the tool returns no results, state:
   "I couldn't find relevant information in your uploaded documents."
4. At the end of your answer, copy the Source citations from the tool result
   exactly as they appear (starting with 'Source:'). Do NOT include any '---CITATIONS---' marker. Do not rephrase citations.
5. Do NOT add information not present in the tool's response. Keep answers concise and grounded.
"""


async def chat_node(state: AgentState) -> Dict[str, Any]:
    try:
        collection_names = state.get("collection_names")
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "default")
        messages = list(state.get("messages", []))

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Check if the latest message is a ToolMessage (meaning a tool just finished executing)
        last_is_tool_result = isinstance(messages[-1], ToolMessage)

        llm = _build_llm()

        if last_is_tool_result:
            # Turn 2: The tool already returned document context.
            # Call the raw LLM without tools so it writes the final grounded answer cleanly.
            logging.info("chat_node generating final grounded answer from tool results")
            await llm_rate_limiter.aacquire()
            response = await llm.ainvoke(messages)
        else:
            # Turn 1: Force retrieval tool execution so general-knowledge hallucinations are impossible.
            last_user_text = messages[-1].content if messages else ""
            if isinstance(last_user_text, list):
                last_user_text = extract_message_text(last_user_text)

            is_summary = is_summary_request(str(last_user_text))
            forced_tool_name = "summarize_document" if is_summary else "rag_query"

            logging.info("chat_node forcing tool call: %s", forced_tool_name)
            await llm_rate_limiter.aacquire()

            try:
                # Standard OpenAI/Groq tool choice function object syntax
                llm_with_tools = llm.bind_tools(
                    AVAILABLE_TOOLS,
                    tool_choice={"type": "function", "function": {"name": forced_tool_name}},
                )
                response = await llm_with_tools.ainvoke(messages)
            except Exception as tool_err:
                logging.warning("Explicit tool_choice failed (%s); falling back to auto", tool_err)
                llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS, tool_choice="auto")
                response = await llm_with_tools.ainvoke(messages)

        if isinstance(response.content, list):
            response.content = extract_message_text(response.content)

        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                if not tc["args"].get("collection_names"):
                    tc["args"]["collection_names"] = collection_names
                tc["args"]["session_id"] = session_id
                tc["args"]["user_id"] = user_id
            tool_names = [tc.get("name") for tc in response.tool_calls]
            logging.info("Calling tool(s): %s | scope: %s", tool_names, collection_names)
        else:
            logging.info("Final answer generated (%d chars)", len(str(response.content)))

        return {"messages": [response]}

    except Exception as e:
        logging.exception("chat_node failed")
        raise CustomException(e, sys)


def should_continue(state: AgentState) -> str:
    """Conditional edge: check if the latest message contains tool calls."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


tools_node = ToolNode(tools=AVAILABLE_TOOLS)
