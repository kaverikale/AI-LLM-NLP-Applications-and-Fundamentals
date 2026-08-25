"""
tools.py — the agent's hands.

A tiny mock "orders" database plus two callable tools, described in
Anthropic's tool-use schema so the planner can call them by name. Also
the `tool_node` function itself — the LangGraph node that actually
executes whatever the planner asked for.

Week 4 addition: search_policies, backed by rag.py's retriever. It's
registered exactly like the order tools below — same TOOLS_SCHEMA
list, same TOOL_FUNCTIONS dict, same tool_node execution path. That's
the point of Lecture 4's "RAG is not replacing the agent, RAG becomes
another capability the agent can use" — tool_node doesn't know or
care that this one's backed by a vector search instead of a dict
lookup.

Swap ORDERS for a real DB call later; nothing else in this file (or
in planner.py / graph.py) needs to change.
"""

import json
from datetime import date

from rag import search_policies

# --- mock data (stand-in for a real orders DB) -------------------------

ORDERS = {
    "A1029": {
        "status": "delivered",
        "ordered_on": "2025-01-03",
        "delivered_on": "2025-01-07",
        "items": ["Bamboo Cutting Board", "Set of 4 Linen Napkins"],
        "total": "$54.00",
    },
    "A1031": {
        "status": "in_transit",
        "ordered_on": "2025-01-10",
        "estimated_delivery": "2025-01-14",
        "items": ["Ceramic Pour-Over Coffee Set"],
        "total": "$38.00",
    },
}

RETURN_WINDOW_DAYS = 30


# --- tool implementations -----------------------------------------------

def get_order_status(order_id: str) -> dict:
    order_id = order_id.upper().lstrip("#")
    order = ORDERS.get(order_id)
    if not order:
        return {"error": f"No order found with ID {order_id}."}
    return {"order_id": order_id, **order}


def check_return_eligibility(order_id: str) -> dict:
    order_id = order_id.upper().lstrip("#")
    order = ORDERS.get(order_id)
    if not order:
        return {"error": f"No order found with ID {order_id}."}
    if order["status"] != "delivered":
        return {
            "eligible": False,
            "reason": "Order hasn't been delivered yet, so a return can't be started.",
        }
    delivered = date.fromisoformat(order["delivered_on"])
    days_since = (date.today() - delivered).days
    return {
        "eligible": days_since <= RETURN_WINDOW_DAYS,
        "days_since_delivery": days_since,
        "return_window_days": RETURN_WINDOW_DAYS,
    }


TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "check_return_eligibility": check_return_eligibility,
    "search_policies": search_policies,
}


# --- Anthropic tool-use schemas ------------------------------------------

TOOLS_SCHEMA = [
    {
        "name": "get_order_status",
        "description": "Look up the current status, contents, and dates for a customer's order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID, e.g. 'A1029'"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": "Check whether an order is still inside the 30-day return window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID, e.g. 'A1029'"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policies",
        "description": (
            "Search Daily Essentials' policy documents — returns, refunds, shipping, "
            "warranty, exchanges — for information relevant to a customer's question. "
            "Use this for any policy or 'how does X work' question instead of guessing; "
            "don't invent return windows, warranty lengths, or refund timelines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, e.g. 'warranty policy for a damaged mixer bought on sale'",
                },
            },
            "required": ["query"],
        },
    },
]


# --- the graph node -------------------------------------------------------

def tool_node(state: dict) -> dict:
    """
    LangGraph node: reads the planner's last message, executes every
    tool_use block it finds, and returns tool_result messages as a
    partial state update. Multiple tool calls in one planner turn are
    all executed and returned together — this is what lets a single
    turn call get_order_status AND search_policies side by side, per
    Lecture 4's "Tool + RAG" worked example.
    """

    last_message = state["messages"][-1]
    tool_use_blocks = [
        block for block in last_message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]

    results = []
    for block in tool_use_blocks:
        fn = TOOL_FUNCTIONS.get(block["name"])
        output = fn(**block["input"]) if fn else {"error": f"Unknown tool '{block['name']}'"}
        results.append({
            "type": "tool_result",
            "tool_use_id": block["id"],
            "content": json.dumps(output),
        })

    return {
        "messages": [{"role": "user", "content": results}],
        "tool_calls_made": state.get("tool_calls_made", 0) + 1,
    }