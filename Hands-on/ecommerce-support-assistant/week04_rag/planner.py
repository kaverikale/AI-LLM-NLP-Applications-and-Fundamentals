

"""
planner.py — the agent's reasoning core.

One node, one job: look at the conversation (+ working memory), and
either answer directly or ask for a tool. Per Lecture 3's Module 8
mapping — Planner -> Node, Decision -> Conditional Edge — the routing
logic (`should_continue`) is deliberately plain Python, used by
graph.py as the conditional edge's predicate.
"""

from llm_client import call_llm
from memory import update_working_memory, memory_to_context
from tools import TOOLS_SCHEMA

MAX_TOOL_HOPS = 4  # loop guard: see the `at_cap` check below

BASE_SYSTEM_PROMPT = """You are a customer support agent for an online store
called Daily Essentials. Answer customer questions about orders, shipping,
returns, and warranty.

You have two kinds of tools:
- get_order_status / check_return_eligibility, for structured data about a
  specific order — use these when a question depends on an actual order.
- search_policies, for company policy documents (returns, refunds, shipping,
  warranty, exchanges) — use this for "what's the policy on X" questions.

A single question can need both — e.g. an order that's damaged or stopped
working may need get_order_status AND search_policies together to answer
correctly. Don't invent order details, dates, tracking numbers, return
windows, or warranty lengths; look them up.

If you already have everything you need (from earlier tool results, or
from what the customer told you earlier in this conversation), answer
directly instead of calling a tool again."""

def planner_node(state: dict) -> dict:
    """LangGraph node: reads state, returns a partial state update."""
    latest_user_text = _latest_user_text(state["messages"])
    memory = update_working_memory(state.get("memory") or {}, latest_user_text)

    system = BASE_SYSTEM_PROMPT
    context = memory_to_context(memory)
    if context:
        system += "\n\n" + context

    # Once we've hit the loop guard, stop offering tools at all — this
    # forces a text-only answer instead of leaving a dangling tool_use
    # with no room left to execute it.
    at_cap = state.get("tool_calls_made", 0) >= MAX_TOOL_HOPS

    response = call_llm(
        messages=state["messages"],
        system=system,
        tools=None if at_cap else TOOLS_SCHEMA,
    )

    return {
        "messages": [{"role": "assistant", "content": _serialize_content(response.content)}],
        "memory": memory,
    }


def should_continue(state: dict) -> bool:
    """Conditional-edge predicate: did the planner's last message ask
    for a tool call?"""
    print("in should_continue", state)
    last_message = state["messages"][-1]
    return any(
        isinstance(block, dict) and block.get("type") == "tool_use"
        for block in last_message["content"]
    )


def _serialize_content(content_blocks) -> list[dict]:
    """The Anthropic SDK returns typed content-block objects; convert
    them to plain dicts so state stays JSON-friendly and matches the
    format tool_node/should_continue already expect."""
    print("in _serialize_content", content_blocks)
    serialized = []
    for block in content_blocks:
        if block.type == "text":
            serialized.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            serialized.append({
                "type": "tool_use", "id": block.id, "name": block.name, "input": block.input,
            })
    return serialized


def _latest_user_text(messages: list[dict]) -> str:
    """Pull plain text out of the most recent *human* turn. After a
    tool round-trip, a 'user' message is actually a list of
    tool_result blocks, not text — skip those."""
    print("in _latest_user_text", messages)

    for message in reversed(messages):
        if message["role"] == "user" and isinstance(message["content"], str):
            return message["content"]