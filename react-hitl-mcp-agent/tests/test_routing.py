"""
Tests for agent/routing.py
===========================
Routing functions are pure — they take state dicts and return strings.
No LLM, no graph, no async needed.

Covers:
  - route_after_guardrail: allowed / blocked branches
  - route_after_agent: no tool calls / safe tools / dangerous tools
  - route_after_human_review: confirmed / rejected / fallback
"""
import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent.routing import (
    route_after_guardrail,
    route_after_agent,
    route_after_human_review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_with_tool_calls(names: list[str]) -> AIMessage:
    """Return an AIMessage that looks like it has tool_calls."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": n, "id": f"call_{i}", "args": {}} for i, n in enumerate(names)]
    return msg


def _ai_text(content: str = "Here is your answer.") -> AIMessage:
    """Return a plain AIMessage with no tool calls."""
    msg = AIMessage(content=content)
    msg.tool_calls = []
    return msg


# ---------------------------------------------------------------------------
# route_after_guardrail
# ---------------------------------------------------------------------------

class TestRouteAfterGuardrail:
    def test_blocked_goes_to_end(self):
        state = {"messages": [], "guardrail_blocked": True}
        assert route_after_guardrail(state) == END

    def test_allowed_goes_to_agent(self):
        state = {"messages": [], "guardrail_blocked": False}
        assert route_after_guardrail(state) == "agent"

    def test_missing_flag_defaults_to_agent(self):
        state = {"messages": []}
        assert route_after_guardrail(state) == "agent"


# ---------------------------------------------------------------------------
# route_after_agent
# ---------------------------------------------------------------------------

class TestRouteAfterAgent:
    def test_no_tool_calls_goes_to_end(self):
        state = {"messages": [_ai_text()]}
        assert route_after_agent(state) == END

    def test_safe_tool_goes_to_tools(self):
        state = {"messages": [_ai_with_tool_calls(["get_order"])]}
        assert route_after_agent(state) == "tools"

    def test_another_safe_tool(self):
        state = {"messages": [_ai_with_tool_calls(["get_customer_history"])]}
        assert route_after_agent(state) == "tools"

    def test_process_refund_goes_to_human_review(self):
        state = {"messages": [_ai_with_tool_calls(["process_refund"])]}
        assert route_after_agent(state) == "human_review"

    def test_cancel_order_goes_to_human_review(self):
        state = {"messages": [_ai_with_tool_calls(["cancel_order"])]}
        assert route_after_agent(state) == "human_review"

    def test_dangerous_tool_takes_priority_over_safe(self):
        """If any tool is dangerous, the whole batch goes to human_review."""
        state = {"messages": [_ai_with_tool_calls(["get_order", "process_refund"])]}
        assert route_after_agent(state) == "human_review"

    def test_multiple_safe_tools_go_to_tools(self):
        state = {"messages": [_ai_with_tool_calls(["get_order", "get_customer_history"])]}
        assert route_after_agent(state) == "tools"


# ---------------------------------------------------------------------------
# route_after_human_review
# ---------------------------------------------------------------------------

class TestRouteAfterHumanReview:
    def test_plain_ai_message_goes_to_end(self):
        """Rejection path: human_review_node added a cancellation AIMessage."""
        state = {"messages": [_ai_text("Action cancelled.")]}
        assert route_after_human_review(state) == END

    def test_confirmed_with_pending_tool_goes_to_tools(self):
        """Confirmation path: pending AIMessage with tool_calls still in history."""
        pending = _ai_with_tool_calls(["process_refund"])
        state = {"messages": [pending, HumanMessage(content="yes")]}
        assert route_after_human_review(state) == "tools"

    def test_no_pending_tools_falls_back_to_agent(self):
        """No tool calls found anywhere → send back to agent."""
        state = {"messages": [HumanMessage(content="yes")]}
        assert route_after_human_review(state) == "agent"

    def test_ai_message_with_tool_calls_goes_to_tools(self):
        """AIMessage that itself has tool_calls → NOT the rejection path."""
        msg = _ai_with_tool_calls(["cancel_order"])
        state = {"messages": [msg]}
        assert route_after_human_review(state) == "tools"
