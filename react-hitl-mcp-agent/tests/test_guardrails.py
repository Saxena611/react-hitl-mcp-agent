"""
Tests for agent/guardrails.py
==============================
Covers:
  - check_message() classification logic and normalization
  - BLOCKED_RESPONSES keys are present
  - InputGuardrail.forward() delegates correctly
  - Provider-level content policy exceptions are treated as injection

All DSPy calls are mocked — no real LLM needed.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — create a fake dspy.Predict result
# ---------------------------------------------------------------------------

def _make_result(decision: str, reason: str = "test reason"):
    result = MagicMock()
    result.decision = decision
    result.reason   = reason
    return result


# ---------------------------------------------------------------------------
# check_message
# ---------------------------------------------------------------------------

class TestCheckMessage:
    """Unit tests for the check_message() classification function."""

    def _call(self, decision_str: str, message: str = "test") -> dict:
        """Patch _get_predict so we control the LLM output."""
        import importlib
        # Reset module-level singleton so each test gets a fresh Predict
        import agent.guardrails as gm
        gm._predict = None

        fake_predict = MagicMock(return_value=_make_result(decision_str))
        with patch.object(gm, "_get_predict", return_value=fake_predict):
            return gm.check_message(message)

    def test_in_scope_allowed(self):
        result = self._call("in_scope")
        assert result["allowed"] is True
        assert result["decision"] == "in_scope"

    def test_out_of_scope_blocked(self):
        result = self._call("out_of_scope")
        assert result["allowed"] is False
        assert result["decision"] == "out_of_scope"

    def test_injection_blocked(self):
        result = self._call("injection")
        assert result["allowed"] is False
        assert result["decision"] == "injection"

    # Normalization — LLMs sometimes produce extra words
    def test_normalizes_out_of_scope_with_extra_text(self):
        result = self._call("out_of_scope (irrelevant topic)")
        assert result["decision"] == "out_of_scope"
        assert result["allowed"] is False

    def test_normalizes_injection_with_extra_text(self):
        result = self._call("injection attempt detected")
        assert result["decision"] == "injection"
        assert result["allowed"] is False

    def test_normalizes_in_scope_with_punctuation(self):
        result = self._call("in_scope.")
        assert result["decision"] == "in_scope"
        assert result["allowed"] is True

    def test_reason_is_included(self):
        import agent.guardrails as gm
        gm._predict = None
        fake_predict = MagicMock(return_value=_make_result("in_scope", "Customer asking about order"))
        with patch.object(gm, "_get_predict", return_value=fake_predict):
            result = gm.check_message("Where is my order?")
        assert result["reason"] == "Customer asking about order"

    def test_content_policy_exception_returns_injection(self):
        import agent.guardrails as gm
        gm._predict = None
        fake_predict = MagicMock(side_effect=Exception("content_filter triggered"))
        with patch.object(gm, "_get_predict", return_value=fake_predict):
            result = gm.check_message("jailbreak attempt")
        assert result["allowed"] is False
        assert result["decision"] == "injection"

    def test_non_content_policy_exception_propagates(self):
        import agent.guardrails as gm
        gm._predict = None
        fake_predict = MagicMock(side_effect=RuntimeError("network error"))
        with patch.object(gm, "_get_predict", return_value=fake_predict):
            with pytest.raises(RuntimeError, match="network error"):
                gm.check_message("anything")


# ---------------------------------------------------------------------------
# InputGuardrail
# ---------------------------------------------------------------------------

class TestInputGuardrail:
    """InputGuardrail is a thin wrapper — verify it delegates to check_message."""

    def test_forward_delegates_to_check_message(self):
        import agent.guardrails as gm
        expected = {"allowed": True, "decision": "in_scope", "reason": "ok"}
        with patch.object(gm, "check_message", return_value=expected) as mock_check:
            guardrail = gm.InputGuardrail()
            result = guardrail.forward(message="Where is my order?")
        mock_check.assert_called_once_with("Where is my order?")
        assert result == expected

    def test_callable_delegates_to_check_message(self):
        import agent.guardrails as gm
        expected = {"allowed": False, "decision": "injection", "reason": "attempt"}
        with patch.object(gm, "check_message", return_value=expected) as mock_check:
            guardrail = gm.InputGuardrail()
            result = guardrail("jailbreak")
        mock_check.assert_called_once_with("jailbreak")
        assert result == expected

    def test_get_guardrail_returns_singleton(self):
        import agent.guardrails as gm
        gm._guardrail = None
        g1 = gm.get_guardrail()
        g2 = gm.get_guardrail()
        assert g1 is g2


# ---------------------------------------------------------------------------
# BLOCKED_RESPONSES
# ---------------------------------------------------------------------------

class TestBlockedResponses:
    """BLOCKED_RESPONSES must cover all known decision values."""

    def test_out_of_scope_key_exists(self):
        from agent.guardrails import BLOCKED_RESPONSES
        assert "out_of_scope" in BLOCKED_RESPONSES
        assert BLOCKED_RESPONSES["out_of_scope"]   # non-empty string

    def test_injection_key_exists(self):
        from agent.guardrails import BLOCKED_RESPONSES
        assert "injection" in BLOCKED_RESPONSES
        assert BLOCKED_RESPONSES["injection"]

    def test_responses_do_not_reveal_mechanism(self):
        """Guard against accidentally leaking guardrail info to attackers."""
        from agent.guardrails import BLOCKED_RESPONSES
        for response in BLOCKED_RESPONSES.values():
            lowered = response.lower()
            assert "guardrail" not in lowered
            assert "injection" not in lowered
            assert "filter" not in lowered
