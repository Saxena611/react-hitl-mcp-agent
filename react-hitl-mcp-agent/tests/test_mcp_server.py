"""
Tests for mcp_server.py
========================
Tests the MCP tool implementations directly — no LLM, no MCP protocol needed.
We call the underlying functions that the @mcp.tool() decorators wrap.

Covers:
  - get_order: found / not found
  - get_customer_history: found / not found
  - process_refund: success / amount exceeds total / order not found
  - cancel_order: processing → success / shipped → error / not found
"""
import json
import sys
import os

import pytest

# The MCP server functions are decorated with @mcp.tool() which makes them
# callables. We can call them directly as regular Python functions.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import mcp_server as srv


# ---------------------------------------------------------------------------
# get_order
# ---------------------------------------------------------------------------

class TestGetOrder:
    def test_existing_order_returns_json(self):
        raw = srv.get_order("ORD-001")
        data = json.loads(raw)
        assert data["order_id"] == "ORD-001"
        assert data["status"] == "delivered"
        assert data["total"] == 899.99

    def test_existing_processing_order(self):
        raw = srv.get_order("ORD-002")
        data = json.loads(raw)
        assert data["status"] == "processing"

    def test_existing_shipped_order(self):
        raw = srv.get_order("ORD-003")
        data = json.loads(raw)
        assert data["status"] == "shipped"
        assert "tracking" in data

    def test_missing_order_returns_error(self):
        raw = srv.get_order("ORD-999")
        data = json.loads(raw)
        assert "error" in data
        assert "ORD-999" in data["error"]

    def test_empty_string_returns_error(self):
        raw = srv.get_order("")
        data = json.loads(raw)
        assert "error" in data


# ---------------------------------------------------------------------------
# get_customer_history
# ---------------------------------------------------------------------------

class TestGetCustomerHistory:
    def test_existing_customer_returns_json(self):
        raw = srv.get_customer_history("CUST-42")
        data = json.loads(raw)
        assert data["customer_id"] == "CUST-42"
        assert data["tier"] == "gold"
        assert isinstance(data["refund_count_12mo"], int)

    def test_missing_customer_returns_error(self):
        raw = srv.get_customer_history("CUST-999")
        data = json.loads(raw)
        assert "error" in data
        assert "CUST-999" in data["error"]


# ---------------------------------------------------------------------------
# process_refund
# ---------------------------------------------------------------------------

class TestProcessRefund:
    def test_valid_refund_returns_refund_id(self):
        raw = srv.process_refund("ORD-001", 899.99, "defective")
        data = json.loads(raw)
        assert "refund_id" in data
        assert data["refund_id"].startswith("REF-")
        assert data["status"] == "processing"
        assert data["amount"] == 899.99

    def test_partial_refund_is_allowed(self):
        raw = srv.process_refund("ORD-001", 500.00, "changed_mind")
        data = json.loads(raw)
        assert "refund_id" in data
        assert data["amount"] == 500.00

    def test_refund_exceeds_total_returns_error(self):
        raw = srv.process_refund("ORD-001", 9999.99, "defective")
        data = json.loads(raw)
        assert "error" in data

    def test_refund_on_missing_order_returns_error(self):
        raw = srv.process_refund("ORD-999", 100.00, "defective")
        data = json.loads(raw)
        assert "error" in data

    def test_refund_id_is_deterministic(self):
        """Same inputs → same refund ID (hash-based)."""
        r1 = json.loads(srv.process_refund("ORD-001", 899.99, "defective"))
        r2 = json.loads(srv.process_refund("ORD-001", 899.99, "defective"))
        assert r1["refund_id"] == r2["refund_id"]

    def test_refund_id_changes_with_different_amount(self):
        r1 = json.loads(srv.process_refund("ORD-001", 899.99, "defective"))
        r2 = json.loads(srv.process_refund("ORD-001", 500.00, "defective"))
        assert r1["refund_id"] != r2["refund_id"]


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    def test_processing_order_can_be_cancelled(self):
        raw = srv.cancel_order("ORD-002", "changed mind")
        data = json.loads(raw)
        assert data["status"] == "cancelled"
        assert data["order_id"] == "ORD-002"
        assert "No charge" in data["message"]

    def test_shipped_order_cannot_be_cancelled(self):
        raw = srv.cancel_order("ORD-003", "changed mind")
        data = json.loads(raw)
        assert "error" in data
        assert "shipped" in data["error"]

    def test_delivered_order_cannot_be_cancelled(self):
        raw = srv.cancel_order("ORD-001", "changed mind")
        data = json.loads(raw)
        assert "error" in data
        assert "delivered" in data["error"]

    def test_missing_order_returns_error(self):
        raw = srv.cancel_order("ORD-999", "mistake")
        data = json.loads(raw)
        assert "error" in data
