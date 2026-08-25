"""
llm_client.py — the thinnest possible wrapper around the LLM API.

Week 1's version took no `tools` or `system` overrides because there
was nothing to configure — one fixed prompt, no function calling.
Week 2 needs both, so call_llm now accepts them. Everything else is
identical: still one client, still one model constant, still no
knowledge in here of planners, tools, or graphs. Orchestration lives
in graph.py / planner.py; this file just talks to the API.
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a customer support assistant for an online store
called Daily Essentials. Answer customer questions about orders, shipping,
returns, and warranty as helpfully as you can."""


def call_llm(
    messages,
    system,
    tools,
    max_tokens=1000,
):
    """
    Returns the *raw* Anthropic response object — not just text —
    because callers now need to inspect it for tool_use blocks, not
    just read a string. See planner.py for how that's used.
    """
    kwargs = dict(model=MODEL, max_tokens=max_tokens, system=system, messages=messages)
    if tools:
        kwargs["tools"] = tools
    return client.messages.create(**kwargs)


def get_response(messages: list[dict]) -> str:
    """Week 1 shape, kept for backward compatibility: no tools, plain
    text in, plain text out."""
    response = call_llm(messages)
    return response.content[0].text