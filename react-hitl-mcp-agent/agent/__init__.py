"""
agent — ReAct + HITL + MCP + Guardrails Agent Package
======================================================

Package layout:

    state.py          AgentState TypedDict, CONFIRMATION_REQUIRED_TOOLS
    prompts.py        SYSTEM_PROMPT (encodes tool workflows)
    guardrails.py     InputGuardrail DSPy module (CheckMessage signature)
    eligibility.py    RefundEligibilityAssessor DSPy module
    providers.py      LLM + DSPy provider detection and construction
    nodes.py          LangGraph node functions
    routing.py        Pure routing functions for conditional edges
    checkpointing.py  SQLite + memory checkpoint backend management
    graph.py          build_graph() — assembles and compiles the StateGraph
    session.py        AgentSession — high-level chat interface

Entry points for external callers:
"""
from .checkpointing import get_db_path, memory_checkpointer, sqlite_checkpointer
from .graph import build_graph
from .session import AgentSession
from .state import CONFIRMATION_REQUIRED_TOOLS, AgentState

__all__ = [
    "AgentSession",
    "build_graph",
    "AgentState",
    "CONFIRMATION_REQUIRED_TOOLS",
    "sqlite_checkpointer",
    "memory_checkpointer",
    "get_db_path",
]
