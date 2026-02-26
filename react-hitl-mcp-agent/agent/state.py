"""
Agent State
===========
Defines the shared state TypedDict that flows through every node in the graph.

add_messages is a reducer: new messages are APPENDED to the list rather than
replacing it, so the LLM sees the full conversation history at every step.
"""
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    # Set True by guardrail_node when a message is blocked.
    # route_after_guardrail reads this flag to skip the agent entirely.
    guardrail_blocked: bool


# Tools that require explicit human confirmation before execution.
# Add any new irreversible tool name here — no routing logic changes needed.
CONFIRMATION_REQUIRED_TOOLS: frozenset[str] = frozenset({"process_refund", "cancel_order"})
