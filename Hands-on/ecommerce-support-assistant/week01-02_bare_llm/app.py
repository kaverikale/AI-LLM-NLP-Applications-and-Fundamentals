"""
Week 1 demo — a bare LLM chatbot, no tools, no data, no memory beyond
the visible chat history.

Run with: streamlit run app.py
"""

import streamlit as st
from llm_client import get_response

st.set_page_config(page_title="Daily Essentials — Support", page_icon="🛒")

st.title("🛒 Daily Essentials — Customer Support")
st.caption("Week 1 build: a raw LLM behind a chat box. Nothing else.")

with st.sidebar:
    st.markdown("### Try asking:")
    st.markdown("""
- "What's your return policy?"
- "What's the status of order **#A1029**?"
- "Can I get a refund on a damaged item after 40 days?"
- "Track my order, ID is **#A1031**"
    """)
    st.markdown("---")
    st.markdown(
        "Note: the policy questions get *plausible-sounding* answers. "
        "The order questions do too — but there's no order data connected "
        "here, so any specific status, date, or tracking number the model "
        "gives you is invented. That's the problem Weeks 3–4 solve."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about an order, return, or policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_response(st.session_state.messages)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
