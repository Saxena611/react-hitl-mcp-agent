"""
Graph Construction
==================
Assembles the LangGraph StateGraph from nodes, edges, and routing functions.

Architecture:

    START
      │
      ▼
    guardrail ──── blocked ──────────────────────► END
      │
      │ allowed
      ▼
    agent ──────────────────────────────────────► END (final answer)
      │ safe tools                                 ▲
      ▼                                            │
    tools ──────────────────────────────────────── ┘  (ReAct loop)

    agent
      │ dangerous tools
      ▼
    human_review ──⚡INTERRUPT──► (user responds) ──► tools (confirmed)
                                                  ──► END  (rejected)

Checkpointer injection:
  build_graph() accepts any LangGraph-compatible checkpointer via the
  `checkpointer` parameter. The caller is responsible for the checkpointer
  lifecycle (opening the connection, calling setup(), closing on shutdown).

  SQLite (durable, default in production):
      async with sqlite_checkpointer() as cp:
          graph = build_graph(tools, checkpointer=cp)

  Memory (ephemeral, default for tests):
      graph = build_graph(tools, checkpointer=memory_checkpointer())

  This separation means graph.py has zero knowledge of which backend is used.
"""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import create_agent_node, guardrail_node, human_review_node
from .providers import build_llm, configure_dspy
from .routing import route_after_agent, route_after_guardrail, route_after_human_review
from .state import AgentState


def build_graph(tools: list, checkpointer: BaseCheckpointSaver | None = None):
    """
    Build and compile the agent graph for the given list of MCP tools.

    Args:
        tools:        LangChain Tool objects from MultiServerMCPClient.get_tools().
        checkpointer: Any LangGraph checkpoint backend. If None, falls back to
                      an in-process MemorySaver (conversations are lost on restart).
                      Pass an AsyncSqliteSaver for durable HITL across restarts.

    Returns:
        A compiled CompiledStateGraph ready for ainvoke() / aget_state() calls.
    """
    configure_dspy()

    llm            = build_llm()
    llm_with_tools = llm.bind_tools(tools)

    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = StateGraph(AgentState)

    workflow.add_node("guardrail",    guardrail_node)
    workflow.add_node("agent",        create_agent_node(llm_with_tools))
    workflow.add_node("tools",        ToolNode(tools))
    workflow.add_node("human_review", human_review_node)

    workflow.set_entry_point("guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"agent": "agent", END: END},
    )
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "human_review": "human_review", END: END},
    )
    workflow.add_edge("tools", "agent")   # ReAct loop
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"tools": "tools", "agent": "agent", END: END},
    )

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
        # Graph PAUSES before human_review. Full state is saved to checkpointer.
        # Resume: ainvoke(None, config) — or ainvoke({messages: cancel_msgs}, config)
    )
