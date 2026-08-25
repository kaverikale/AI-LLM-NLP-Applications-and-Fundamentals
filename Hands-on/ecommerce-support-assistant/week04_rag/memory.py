"""
memory.py — working memory: small, rule-based, no LLM call needed.

This is intentionally the simplest possible version of "memory" — a
regex over the latest message, not a vector store. The point for this
week is the *shape* (extract -> merge -> inject back into context),
not the extraction technique. Swap in something smarter later without
touching planner.py or graph.py.

Note the other kind of memory in this system: LangGraph's checkpointer
(wired up in graph.py) persists the *entire* state — including this
working-memory dict — across turns for a given thread_id. That's
closer to what Lecture 3 called long-term memory: it survives outside
any single node's return value, keyed by conversation thread.
"""

import re

ORDER_ID_PATTERN = re.compile(r"#?([A-Z]\d{4,6})")


def extract_entities(text: str) -> dict:
    """Look for anything worth remembering in a chunk of user text.
    Right now: just order-ID-shaped tokens like '#A1029'."""
    print("in extract_entities", text)
    found = {}
    match = ORDER_ID_PATTERN.search(text.upper())
    if match:
        found["last_order_id"] = match.group(1)
    return found


def update_working_memory(current_memory: dict, latest_user_text: str) -> dict:
    """Merge newly-found entities into what's already known. Existing
    facts are kept unless this turn overwrites them with something new
    (e.g. the user mentions a different order ID)."""
    print("in update_working_memory", current_memory, latest_user_text)
    updated = dict(current_memory)
    updated.update(extract_entities(latest_user_text))
    return updated


def memory_to_context(memory: dict) -> str:
    """Render working memory as a short block the planner's system
    prompt can include, so the model doesn't need the user to repeat
    themselves every turn."""
    print("in memory_to_context", memory)

    if not memory:
        return ""
    lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in memory.items()]
    return "Known context from earlier in this conversation:\n" + "\n".join(lines)