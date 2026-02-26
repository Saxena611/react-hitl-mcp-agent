"""
Refund Eligibility Assessor (DSPy)
===================================
Handles structured policy decisions — a sub-task that needs reliable, typed output.

Design principle: DSPy for sub-reasoning, LangGraph for orchestration.
  - Main LLM: routing, conversation, tool selection
  - DSPy: policy assessment with a full reasoning trace for audits

The assessor is called from within MCP tool implementations, not by the LLM directly.
ChainOfThought adds an explicit reasoning trace before producing output.
"""
import json

import dspy


class AssessEligibility(dspy.Signature):
    """
    Assess refund eligibility based on order details, customer history, and policy.

    Policy rules:
    - Standard return window: 30 days from delivery
    - Gold/Platinum tier: 60-day window
    - Defective items: 90-day window regardless of tier
    - Max 3 refunds per 12 months
    - Changed-mind returns: 20% restocking fee
    """
    order_info:    str = dspy.InputField(desc="Order JSON: status, total, order_date, delivered_date")
    customer_info: str = dspy.InputField(desc="Customer JSON: tier, refund_count_12mo, account_age_days")
    reason:        str = dspy.InputField(desc="Customer's stated reason for the refund request")

    eligible:       bool  = dspy.OutputField(desc="True if eligible under policy")
    explanation:    str   = dspy.OutputField(desc="Clear explanation of the eligibility decision")
    max_amount:     float = dspy.OutputField(desc="Maximum refundable amount in USD (0.0 if not eligible)")
    recommendation: str   = dspy.OutputField(
        desc="Best action: full_refund | partial_refund | store_credit | replacement | denial"
    )


class RefundEligibilityAssessor(dspy.Module):
    """
    Produces a structured eligibility verdict with an audit-ready reasoning trace.

    Returns:
        eligible        — bool, True if customer qualifies for a refund
        explanation     — human-readable reasoning
        max_amount      — maximum USD amount eligible for refund
        recommendation  — full_refund | partial_refund | store_credit | replacement | denial
        reasoning_trace — the CoT chain for debugging and compliance audits
    """

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(AssessEligibility)

    def forward(self, order: dict, customer: dict, reason: str) -> dict:
        result = self.assess(
            order_info=json.dumps(order, default=str),
            customer_info=json.dumps(customer),
            reason=reason,
        )
        return {
            "eligible":        result.eligible,
            "explanation":     result.explanation,
            "max_amount":      float(result.max_amount),
            "recommendation":  result.recommendation,
            "reasoning_trace": result.reasoning,
        }
