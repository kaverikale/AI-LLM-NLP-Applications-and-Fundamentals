"""
Week 2 demo — same storefront, now backed by a LangGraph agent: a
planner that can call real tools (order lookup, return-eligibility
check) and working memory that carries context across turns.

Run with: streamlit run app.py
"""

import uuid
import streamlit as st
from graph import graph

st.set_page_config(page_title="Daily Essentials — Support", page_icon="🛒")

st.title("🛒 Daily Essentials — Customer Support")
st.caption("Week 4 build: Planner + Tools + Memory + RAG, wired together with LangGraph.")

with st.sidebar:
    st.markdown("### Try asking:")
    st.markdown("""
- "What's the status of order **#A1029**?"
- "What's your warranty policy on mixers?"
- "I bought a mixer in the Diwali sale and it stopped working after 20 days — can I return it or get a replacement?"
- "Track my order, ID is **#A1031**" — then follow up with *"can I return it?"* with no ID repeated
    """)
    st.markdown("---")
    st.markdown(
        "Order questions hit a real (mock) database through a tool call. Policy "
        "questions — returns, refunds, shipping, warranty, exchanges — now hit a "
        "real (mock) knowledge base through RAG: your question gets embedded, "
        "compared against indexed policy chunks, and the top matches are handed "
        "to the model as context. Watch 'Tool activity' below each answer — "
        "`search_policies` shows up there just like the order tools."
    )
    st.markdown(
        "_First run may take a few seconds while the local embedding model "
        "downloads._"
    )
    if st.button("Clear conversation"):
        for key in ("messages", "thread_id"):
            st.session_state.pop(key, None)
        st.rerun()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # display-only transcript

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _extract_reply(messages: list[dict]) -> tuple[str, list[str]]:
    """Pull the final text answer and this turn's tool names out of
    the graph's full message history, for display purposes only."""
    final_text = "".join(
        block["text"] for block in messages[-1]["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )

    tools_used = []
    for msg in reversed(messages[:-1]):
        if msg["role"] == "user" and isinstance(msg["content"], str):
            break  # reached the human message that started this turn
        if msg["role"] == "assistant":
            tools_used.extend(
                block["name"] for block in msg["content"]
                if isinstance(block, dict) and block.get("type") == "tool_use"
            )

    return final_text or "(no response)", list(reversed(tools_used))


if prompt := st.chat_input("Ask about an order, return, or policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke(
                {"messages": [{"role": "user", "content": prompt}], "tool_calls_made": 0},
                config=config,
            )
            reply_text, tools_used = _extract_reply(result["messages"])

        st.markdown(reply_text)
        if tools_used:
            st.caption(f"🔧 Tool activity: {', '.join(tools_used)}")

    st.session_state.messages.append({"role": "assistant", "content": reply_text})