import json
import sys
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.exception import CustomException
from src.graph.agent_nodes import chat_node, should_continue, tools_node
from src.graph.helpers import extract_message_text
from src.graph.state import AgentState
from src.logger import logging


def build_react_agent():
    """Assembles and compiles the ReAct agent graph with tool-calling.

    Note: Execution is stateless per-request because conversation history is already
    persisted and managed via PostgreSQL MemoryManager.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("chat_node", chat_node)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "chat_node")
    workflow.add_conditional_edges(
        "chat_node",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "chat_node")

    return workflow.compile()


# Compiled ReAct agent
react_agent = build_react_agent()


async def astream_agent_response(
    query: str,
    chat_history: List[BaseMessage],
    collection_names: Optional[List[str]] = None,
    session_id: str = "default",
    user_id: str = "default",
) -> AsyncGenerator[Dict[str, Any], None]:
    """Streams token chunks and tool status updates from the ReAct agent."""
    try:
        messages = list(chat_history) + [HumanMessage(content=query)]
        input_state: AgentState = {
            "messages": messages,
            "collection_names": collection_names,
            "session_id": session_id,
            "user_id": user_id,
        }
        config = {
            "run_name": f"react_agent_{session_id[:8]}",
            "tags": [f"user_{user_id}", f"session_{session_id[:8]}"],
        }

        logging.info("Starting ReAct agent streaming | session: %s", session_id)

        full_answer_parts: List[str] = []
        is_tool_running = False

        async for event in react_agent.astream_events(input_state, config=config, version="v2"):
            kind = event.get("event")

            if kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                is_tool_running = True
                logging.info("ReAct streaming event: tool_start | %s", tool_name)
                yield {
                    "type": "tool_start",
                    "name": tool_name,
                    "input": event.get("data", {}).get("input", {}),
                }

            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                is_tool_running = False
                logging.info("ReAct streaming event: tool_end | %s", tool_name)
                yield {
                    "type": "tool_end",
                    "name": tool_name,
                }

            elif kind == "on_chat_model_stream":
                # Only stream tokens when the model is generating the final user-facing reply
                # (not during intermediate tool-calling decision steps)
                chunk = event.get("data", {}).get("chunk")
                if chunk and not is_tool_running:
                    # Ignore tool call argument chunks
                    if getattr(chunk, "tool_call_chunks", None):
                        continue

                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    text = extract_message_text(content)
                    if text:
                        full_answer_parts.append(text)
                        yield {
                            "type": "token",
                            "content": text,
                        }

        final_answer = "".join(full_answer_parts).strip()
        logging.info("ReAct agent streaming completed | answer length: %d", len(final_answer))
        yield {
            "type": "done",
            "final_answer": final_answer,
        }

    except Exception as e:
        logging.exception("astream_agent_response encountered error")
        raise CustomException(e, sys)
