"""
llm_client.py — the thinnest possible wrapper around the LLM API.

This is intentional: Week 1's whole point is showing what an LLM does
with *nothing* around it — no tools, no retrieved data, no memory beyond
the raw chat history. Every later week adds a layer in front of or
around this call; this file itself barely changes.
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


def get_response(messages: list[dict]) -> str:
    """
    messages: list of {"role": "user" | "assistant", "content": str}
    Returns the assistant's reply as plain text.

    Note what's NOT happening here: no order lookup, no policy document,
    no tool calls. The model answers purely from what it was trained on
    plus whatever is in the conversation so far. That's the demo.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text
