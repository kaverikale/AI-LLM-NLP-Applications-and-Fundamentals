# Week 1 — Bare LLM Application

## What this demonstrates
An LLM application with the absolute minimum wrapper: a chat UI and an
API call. No retrieval, no tools, no order data — even though
`data/orders.json` already exists one folder up.

That gap is the point. Students should leave this demo believing "an LLM
alone isn't enough," which sets up Weeks 3 and 4.

## Run it
```bash
cd week01_bare_llm
streamlit run app.py
```

## Suggested live-demo script

1. **Ask a policy question first** — *"What's your return window?"*
   The model answers reasonably (it's general knowledge about how return
   policies usually work) — but it is guessing, not reading Northwind's
   actual policy. Flag this for later ("looks right, but is it grounded in
   *our* policy? We can't tell.").

2. **Ask an order-status question** — *"What's the status of order #A1029?"*
   The model will produce a confident, specific-sounding answer (a status,
   maybe a date) with **zero connection to `data/orders.json`**. Open that
   file side-by-side and show the class the real record — or that the order
   ID doesn't even matter, the model will happily answer for a made-up ID
   too.

3. **Ask the same question with a different, non-existent order ID** —
   e.g. *"Status of order #ZZZ999?"* — to show it doesn't know the
   difference between a real and fake order. This is the clearest
   hallucination moment.

4. **Bridge to Week 3**: "The model isn't broken — it's doing exactly what
   it was trained to do: predict a plausible-sounding continuation. It has
   no way to *check* anything. Next week we give it that ability."

## Code walkthrough points
- `llm_client.py` — show how thin this is: system prompt + message history,
  nothing else. This is the baseline every later week adds to.
- `app.py` — point out `st.session_state.messages` as the *only* memory —
  it's just the visible conversation, not true persistent memory (that
  distinction matters again in Week 6/13).
