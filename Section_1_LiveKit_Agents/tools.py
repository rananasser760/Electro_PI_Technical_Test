"""
Shared business logic for the food-delivery support agent's tools.

Kept separate from agent.py / console_demo.py so the SAME implementation
is used whether the tool is invoked through the real LiveKit
`@function_tool` pipeline or through the text-based console demo's
manual function-calling loop. This is also what makes "add a second
tool safely" easy: one function here, one schema entry, done.
"""

from __future__ import annotations
import random

# ---------------------------------------------------------------------
# Mock "database"
# ---------------------------------------------------------------------
_ORDERS = {
    "A1001": {"status": "out_for_delivery", "eta_minutes": 12, "restaurant": "Sushi Go"},
    "A1002": {"status": "preparing", "eta_minutes": 25, "restaurant": "Burger Barn"},
    "A1003": {"status": "delivered", "eta_minutes": 0, "restaurant": "Pasta Place"},
    "A1004": {"status": "cancelled", "eta_minutes": None, "restaurant": "Taco Town"},
}

# Orders that are allowed to be cancelled (business rule used below)
_CANCELLABLE_STATUSES = {"preparing", "confirmed"}


# ---------------------------------------------------------------------
# Tool 1 (required): get_order_status
# ---------------------------------------------------------------------
def get_order_status_impl(order_id: str) -> str:
    """Look up an order and return a short natural-language status string."""
    order_id = order_id.strip().upper()
    order = _ORDERS.get(order_id)

    if order is None:
        # Deliberate, structured error -> the LLM gets a clear signal to
        # relay to the user instead of hallucinating a status.
        return f"error: no order found with id '{order_id}'"

    if order["status"] == "delivered":
        return f"Order {order_id} from {order['restaurant']} has already been delivered."
    if order["status"] == "cancelled":
        return f"Order {order_id} was cancelled and is not being delivered."
    return (
        f"Order {order_id} from {order['restaurant']} is currently '{order['status']}', "
        f"estimated delivery in {order['eta_minutes']} minutes."
    )


GET_ORDER_STATUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Get the current status and ETA for a food delivery order by its order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. 'A1001'.",
                }
            },
            "required": ["order_id"],
        },
    },
}


# ---------------------------------------------------------------------
# Tool 2 (added to demonstrate the write-up's "second tool" guidance):
# cancel_order
# ---------------------------------------------------------------------
def cancel_order_impl(order_id: str, reason: str) -> str:
    """Attempt to cancel an order. Demonstrates validation + failure paths."""
    order_id = order_id.strip().upper()
    order = _ORDERS.get(order_id)

    if order is None:
        return f"error: no order found with id '{order_id}'"

    if order["status"] not in _CANCELLABLE_STATUSES:
        return (
            f"error: order {order_id} cannot be cancelled because it is already "
            f"'{order['status']}'."
        )

    # Simulate a flaky downstream service (~1 in 5 calls) so the demo can
    # show the agent handling a genuine tool failure, not just bad input.
    if random.random() < 0.2:
        return "error: cancellation service temporarily unavailable, please try again"

    order["status"] = "cancelled"
    return f"Order {order_id} has been cancelled. Reason logged: '{reason}'."


CANCEL_ORDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "cancel_order",
        "description": "Cancel a food delivery order if it hasn't shipped yet.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. 'A1002'."},
                "reason": {"type": "string", "description": "Why the customer wants to cancel."},
            },
            "required": ["order_id", "reason"],
        },
    },
}

ALL_SCHEMAS = [GET_ORDER_STATUS_SCHEMA, CANCEL_ORDER_SCHEMA]
IMPLS = {
    "get_order_status": get_order_status_impl,
    "cancel_order": cancel_order_impl,
}
