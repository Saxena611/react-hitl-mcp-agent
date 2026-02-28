"""
MCP Server: ShopEasy Order Service
===================================
Exposes order management tools via the Model Context Protocol.

Tools:
  - get_order           → Read-only. Look up any order.
  - get_customer_history → Read-only. Get customer profile and history.
  - process_refund      → WRITE. Requires human confirmation in the agent.
  - cancel_order        → WRITE. Requires human confirmation in the agent.

Run standalone:   python mcp_server.py
Or via agent:     the agent starts this as a subprocess (stdio transport).
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ShopEasy Order Service")

# ---------------------------------------------------------------------------
# In-memory data store — replace with real DB calls in production
# ---------------------------------------------------------------------------

ORDERS = {
    "ORD-001": {
        "order_id": "ORD-001",
        "customer_id": "CUST-42",
        "status": "delivered",
        "items": [{"name": "Laptop Pro X", "qty": 1, "price": 899.99}],
        "total": 899.99,
        "order_date": "2026-01-15",
        "delivered_date": "2026-01-20",
        "carrier": "FedEx",
        "tracking": "794644823724",
    },
    "ORD-002": {
        "order_id": "ORD-002",
        "customer_id": "CUST-42",
        "status": "processing",
        "items": [
            {"name": "Wireless Mouse", "qty": 2, "price": 29.99},
            {"name": "USB-C Hub", "qty": 1, "price": 49.99},
        ],
        "total": 109.97,
        "order_date": "2026-02-20",
    },
    "ORD-003": {
        "order_id": "ORD-003",
        "customer_id": "CUST-42",
        "status": "shipped",
        "items": [{"name": "Mechanical Keyboard", "qty": 1, "price": 149.99}],
        "total": 149.99,
        "order_date": "2026-02-18",
        "carrier": "UPS",
        "tracking": "1Z999AA10123456784",
        "estimated_delivery": "2026-02-25",
    },
}

CUSTOMERS = {
    "CUST-42": {
        "customer_id": "CUST-42",
        "name": "Alex Johnson",
        "email": "alex@example.com",
        "tier": "gold",
        "account_age_days": 540,
        "total_spend": 4250.00,
        "refund_count_12mo": 0,
        "orders": ["ORD-001", "ORD-002", "ORD-003"],
    }
}


# ---------------------------------------------------------------------------
# Read-only tools — no confirmation needed
# ---------------------------------------------------------------------------

@mcp.tool()
def get_customer_orders(customer_id: str) -> str:
    """
    Get a full summary of all orders for a customer in a single call.

    Returns each order's ID, status, items, total, and key dates.
    Call this when the user doesn't reference a specific order, or when you
    need the full picture to proactively surface relevant actions
    (e.g., a delivered order eligible for refund, a processing order still
    cancellable, a shipped order with a delayed estimated delivery).

    Args:
        customer_id: The customer's unique identifier
    """
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return json.dumps({"error": f"Customer '{customer_id}' not found."})

    orders = []
    for order_id in customer.get("orders", []):
        order = ORDERS.get(order_id)
        if order:
            orders.append({
                "order_id":           order["order_id"],
                "status":             order["status"],
                "total":              order["total"],
                "order_date":         order.get("order_date"),
                "delivered_date":     order.get("delivered_date"),
                "estimated_delivery": order.get("estimated_delivery"),
                "items":              order["items"],
            })

    return json.dumps({
        "customer_id": customer_id,
        "name":        customer["name"],
        "tier":        customer["tier"],
        "orders":      orders,
    })


@mcp.tool()
def get_order(order_id: str) -> str:
    """
    Retrieve order information by order ID.

    Returns full order details: status, items, total, shipping info.
    Always call this first when a customer references an order number.

    Args:
        order_id: The order identifier (e.g., "ORD-001")
    """
    order = ORDERS.get(order_id)
    if not order:
        return json.dumps({"error": f"Order '{order_id}' not found."})
    return json.dumps(order)


@mcp.tool()
def get_customer_history(customer_id: str) -> str:
    """
    Get a customer's profile, tier, and purchase history.

    Returns tier level, total spend, refund count (past 12 months),
    and account age. Call this before assessing refund eligibility.

    Args:
        customer_id: The customer's unique identifier
    """
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return json.dumps({"error": f"Customer '{customer_id}' not found."})
    return json.dumps(customer)


# ---------------------------------------------------------------------------
# Write tools — the agent's routing logic requires human confirmation
# before these are called (via interrupt_before=["human_review"])
# ---------------------------------------------------------------------------

@mcp.tool()
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """
    Process a financial refund for an order.

    ⚠️  IRREVERSIBLE OPERATION — the agent will pause for human
    confirmation before executing this tool.

    Only call after:
      1. Verifying the order exists (get_order)
      2. Checking customer history (get_customer_history)
      3. Receiving explicit confirmation from the user

    Args:
        order_id: The order to refund (e.g., "ORD-001")
        amount:   Amount in USD (must be <= order total)
        reason:   Reason code: defective | wrong_item | changed_mind | other
    """
    order = ORDERS.get(order_id)
    if not order:
        return json.dumps({"error": f"Order '{order_id}' not found."})
    if amount > order["total"]:
        return json.dumps({"error": f"Refund amount ${amount} exceeds order total ${order['total']}."})

    refund_id = f"REF-{abs(hash(order_id + str(amount))) % 99999:05d}"
    return json.dumps({
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": amount,
        "status": "processing",
        "estimated_credit": "3-5 business days",
        "reason": reason,
    })


@mcp.tool()
def cancel_order(order_id: str, reason: str) -> str:
    """
    Cancel an order that is still in processing status.

    ⚠️  Cannot cancel shipped or delivered orders.
    For those, use process_refund instead.
    The agent will pause for human confirmation before executing this.

    Args:
        order_id: The order to cancel
        reason:   Customer's reason for cancellation
    """
    order = ORDERS.get(order_id)
    if not order:
        return json.dumps({"error": f"Order '{order_id}' not found."})
    if order["status"] != "processing":
        return json.dumps({
            "error": (
                f"Cannot cancel order in '{order['status']}' status. "
                "Use process_refund for delivered orders."
            )
        })
    return json.dumps({
        "order_id": order_id,
        "status": "cancelled",
        "message": "Order cancelled. No charge applied.",
        "reason": reason,
    })


if __name__ == "__main__":
    mcp.run()
