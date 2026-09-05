from langgraph.prebuilt import ToolNode

# Scaffold for future MCP client tools integration.
# ToolNode natively handles async-callable tools provided by an MCP transport.
tool_node = ToolNode(tools=[])
