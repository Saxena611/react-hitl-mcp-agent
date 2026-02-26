"""
Graph Nodes
===========
Each function here is one node in the LangGraph StateGraph.

Node responsibilities:
  guardrail_node    — runs first; blocks off-topic / injection messages
  create_agent_node — the ReAct brain; calls tools or produces a final answer
  human_review_node — processes yes/no after an HITL interrupt

Design principle: nodes are pure state transformers.
They read AgentState, return a dict of updated fields, and never call
graph.invoke() themselves — routing is handled by separate routing functions.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .guardrails import BLOCKED_RESPONSES, get_guardrail
from .prompts import SYSTEM_PROMPT
from .state import AgentState

logger = logging.getLogger(__name__)


def guardrail_node(state: AgentState) -> dict:
    """
    The first node in the graph — runs before the agent on every message.

    Two checks:
      1. Is this message relevant to ShopEasy customer support?
      2. Is this a prompt injection attempt?

    Blocked → injects a canned AIMessage, sets guardrail_blocked=True.
    Allowed → returns empty dict (state unchanged), guardrail_blocked=False.

    Uses dspy.Predict (not ChainOfThought) — runs on every message so speed
    matters more than reasoning depth.
    """
    last_human = next(
        (m for m in reversed(list(state["messages"])) if isinstance(m, HumanMessage)),
        None,
    )

    if last_human is None:
        return {"guardrail_blocked": False}

    result = get_guardrail().forward(message=last_human.content)

    logger.info(
        "[guardrail] decision=%s allowed=%s reason=%r",
        result["decision"], result["allowed"], result["reason"],
    )

    if not result["allowed"]:
        blocked_response = BLOCKED_RESPONSES.get(
            result["decision"], BLOCKED_RESPONSES["out_of_scope"]
        )
        return {
            "messages": [AIMessage(content=blocked_response)],
            "guardrail_blocked": True,
        }

    return {"guardrail_blocked": False}


def create_agent_node(llm_with_tools):
    """
    Factory that returns the agent node bound to a specific LLM + tools.

    The agent node is the ReAct brain: it sees the full conversation history
    and decides to call a tool or produce a final answer.
    Injects the system prompt at position 0 if not already present.
    """
    def agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node


def human_review_node(state: AgentState) -> dict:
    """
    Processes the human's yes/no response after an HITL interrupt.

    This node only runs after graph.invoke() is called again post-interrupt.
    By this point the user has replied and their message is already in state.

    Confirmed → return empty messages; route_after_human_review finds the
                pending tool calls and sends them to "tools".
    Rejected  → inject a cancellation AIMessage; route sends to END.
    Ambiguous → treat as confirmed (fall through to tools); the agent will
                handle unclear responses gracefully.
    """
    last = list(state["messages"])[-1]

    if isinstance(last, HumanMessage):
        content = last.content.lower().strip()

        confirmed_words = {"yes", "confirm", "proceed", "ok", "sure", "go", "yep", "do it"}
        rejected_words  = {"no", "cancel", "stop", "abort", "nope", "don't"}

        if any(w in content for w in confirmed_words):
            return {"messages": []}

        if any(w in content for w in rejected_words):
            return {"messages": [AIMessage(
                content=(
                    "No problem — I've cancelled that action. "
                    "Is there anything else I can help you with?"
                )
            )]}

    return {"messages": []}
