# Week 3 — AI Agents (stub)

## What lands here
A router/agent that reads the user's message and decides which path to take:
- FAQ / policy question -> answer directly (still ungrounded until Week 4)
- Order status question -> parse an order ID, look it up in `data/orders.json`
- Return/refund request -> check eligibility against order date

## Planned approach
Simple explicit routing first (if/else or a classification prompt) *before*
introducing a framework — the point is to show students what an agent
framework (Week 13) is actually automating, not to start with the framework.

Ask for this week's code when you are ready to build it.
