"""
Tests for agent/session.py — pure helper functions
====================================================
Tests is_confirmation(), build_cancel_messages(), and describe_pending()
without touching the async graph or MCP server.

Covers:
  - is_confirmation: all yes/no words, ambiguous input, mixed case
  - build_cancel_messages: correct ToolMessage construction, multiple calls
  - describe_pending: process_refund and cancel_order formatting, fallback
"""
import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from agent.session import (
    build_cancel_messages,
    describe_pending,
    is_confirmation,
)


# ---------------------------------------------------------------------------
# is_confirmation
# ---------------------------------------------------------------------------

class TestIsConfirmation:
    @pytest.mark.parametrize("msg", ["yes", "Yes", "YES", "confirm", "Confirm",
                                      "proceed", "ok", "sure", "go", "yep", "do it"])
    def test_affirmative_words_return_true(self, msg):
        assert is_confirmation(msg) is True

    @pytest.mark.parametrize("msg", ["no", "No", "NO", "cancel", "stop",
                                      "abort", "nope", "don't"])
    def test_negative_words_return_false(self, msg):
        assert is_confirmation(msg) is False

    @pytest.mark.parametrize("msg", ["maybe", "hmm", "what?", "tell me more", ""])
    def test_ambiguous_returns_none(self, msg):
        assert is_confirmation(msg) is None

    def test_leading_trailing_whitespace_stripped(self):
        assert is_confirmation("  yes  ") is True
        assert is_confirmation("  no  ")  is False

    def test_substring_match_works(self):
        """Word detection uses `in` so "please proceed" should be True."""
        assert is_confirmation("please proceed") is True
        assert is_confirmation("please cancel") is False


# ---------------------------------------------------------------------------
# build_cancel_messages
# ---------------------------------------------------------------------------

def _make_state_with_pending(tool_calls: list[dict]) -> MagicMock:
    """Create a fake graph state with an AIMessage that has tool_calls."""
    ai_msg = AIMessage(content="")
    ai_msg.tool_calls = tool_calls

    state = MagicMock()
    state.values = {"messages": [ai_msg]}
    return state


class TestBuildCancelMessages:
    def test_single_tool_call_produces_one_tool_message(self):
        state = _make_state_with_pending([
            {"name": "process_refund", "id": "call_abc123", "args": {}}
        ])
        msgs = build_cancel_messages(state)
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == "call_abc123"
        assert "cancelled" in msgs[0].content.lower()

    def test_multiple_tool_calls_produce_multiple_messages(self):
        state = _make_state_with_pending([
            {"name": "process_refund", "id": "call_1", "args": {}},
            {"name": "cancel_order",   "id": "call_2", "args": {}},
        ])
        msgs = build_cancel_messages(state)
        assert len(msgs) == 2
        ids = {m.tool_call_id for m in msgs}
        assert ids == {"call_1", "call_2"}

    def test_no_pending_tool_calls_returns_empty_list(self):
        state = MagicMock()
        state.values = {"messages": []}
        msgs = build_cancel_messages(state)
        assert msgs == []

    def test_only_most_recent_ai_message_is_used(self):
        """If there are multiple AIMessages, only the last one is cancelled."""
        old_ai = AIMessage(content="")
        old_ai.tool_calls = [{"name": "get_order", "id": "old_call", "args": {}}]

        new_ai = AIMessage(content="")
        new_ai.tool_calls = [{"name": "process_refund", "id": "new_call", "args": {}}]

        state = MagicMock()
        state.values = {"messages": [old_ai, new_ai]}
        msgs = build_cancel_messages(state)
        assert len(msgs) == 1
        assert msgs[0].tool_call_id == "new_call"


# ---------------------------------------------------------------------------
# describe_pending
# ---------------------------------------------------------------------------

class TestDescribePending:
    def test_process_refund_includes_amount_and_order(self):
        pending = {"tool": "process_refund", "args": {"amount": 899.99, "order_id": "ORD-001"}}
        desc = describe_pending(pending)
        assert "$899.99" in desc
        assert "ORD-001" in desc
        assert "yes" in desc.lower() or "proceed" in desc.lower()

    def test_process_refund_without_amount_uses_fallback(self):
        pending = {"tool": "process_refund", "args": {"order_id": "ORD-001"}}
        desc = describe_pending(pending)
        assert "amount" in desc.lower() or "$" in desc

    def test_cancel_order_includes_order_id(self):
        pending = {"tool": "cancel_order", "args": {"order_id": "ORD-002"}}
        desc = describe_pending(pending)
        assert "ORD-002" in desc
        assert "cancel" in desc.lower()

    def test_unknown_tool_uses_generic_message(self):
        pending = {"tool": "some_future_tool", "args": {}}
        desc = describe_pending(pending)
        assert "some_future_tool" in desc
        assert "proceed" in desc.lower()

    def test_refund_description_warns_irreversible(self):
        pending = {"tool": "process_refund", "args": {"amount": 100.0, "order_id": "ORD-001"}}
        desc = describe_pending(pending)
        # Should contain some warning about the action being final
        assert any(w in desc.lower() for w in ["cannot be undone", "irreversible", "transaction"])

    def test_cancel_description_warns_irreversible(self):
        pending = {"tool": "cancel_order", "args": {"order_id": "ORD-002"}}
        desc = describe_pending(pending)
        assert any(w in desc.lower() for w in ["cannot be undone", "irreversible"])
