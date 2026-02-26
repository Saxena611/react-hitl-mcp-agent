"""
Routing Functions
=================
Pure functions that read AgentState and return a destination node name.
LangGraph calls these at conditional edges to decide where execution goes next.

Pure functions = easy to unit-test without spinning up the full graph.

Graph routing map:
  guardrail  → route_after_guardrail  → "agent" | END
  agent      → route_after_agent      → "tools" | "human_review" | END
  human_review → route_after_human_review → "tools" | "agent" | END
"""
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END

from .state import AgentState, CONFIRMATION_REQUIRED_TOOLS


def route_after_guardrail(state: AgentState) -> Literal["agent", "__end__"]:
    """Blocked messages skip the agent entirely and go straight to END."""
    if state.get("guardrail_blocked", False):
        return END
    return "agent"


def route_after_agent(state: AgentState) -> Literal["tools", "human_review", "__end__"]:
    """
    After the agent thinks, decide what happens next:
      - No tool calls  → END (LLM produced a final answer)
      - Safe tool call → "tools" (execute and loop back via ReAct edge)
      - Dangerous tool → "human_review" (triggers interrupt_before pause)
    """
    last = state["messages"][-1]

    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return END

    for tc in last.tool_calls:
        if tc["name"] in CONFIRMATION_REQUIRED_TOOLS:
            return "human_review"

    return "tools"


def route_after_human_review(state: AgentState) -> Literal["tools", "agent", "__end__"]:
    """
    After the human responds to an HITL prompt:
      - Rejected (human_review_node added a plain AIMessage) → END
      - Confirmed (empty messages returned)                  → "tools"
      - Fallback (no pending tool calls found)               → "agent"
    """
    last = state["messages"][-1]

    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        return END

    for msg in reversed(list(state["messages"])):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return "tools"

    return "agent"
