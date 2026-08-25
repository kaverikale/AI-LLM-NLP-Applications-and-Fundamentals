"""
state.py — the shape of memory, literally.

Per Lecture 3's Module 8 mapping: Memory -> State. Everything a node
needs to read or can write lives here. Two different kinds of memory
show up in this one object:

  - `messages`  : the full conversation, in Anthropic's own message
                  format. This is what makes multi-turn tool use work
                  at all — every node sees everything said so far.
  - `memory`    : a small scratch dict for facts worth remembering
                  explicitly (e.g. "the order ID the user mentioned"),
                  separate from the raw transcript. See memory.py.
"""

from typing import Annotated, TypedDict


def append_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """
    Reducer for the `messages` field: new messages a node returns are
    appended, never replace the history. Plain list concatenation —
    no LangChain message objects, just the dicts the Anthropic API
    already understands.
    """
    return left + right


class AgentState(TypedDict):
    # the conversation itself — see append_messages above
    messages: Annotated[list[dict], append_messages]

    # working memory: facts extracted mid-conversation (e.g. an order
    # ID), carried forward turn to turn. No custom reducer — a node
    # that wants to update it just returns the *whole* merged dict,
    # so plain "last write wins" is correct here.
    memory: dict

    # loop guard — counts tool round-trips in the *current* turn so a
    # confused planner can't call tools forever. Reset to 0 by the
    # caller at the start of every new user turn (see app.py).
    tool_calls_made: int