# E-Commerce Support Assistant — Course Project

One running system, built up week by week across "NLP, LLMs & AI Applications
(Top-Down Approach)". Every lecture adds one more layer to the *same* assistant,
so students see cause and effect instead of a new toy demo each week.

## The problem statement

> Build an AI-powered customer support assistant for an e-commerce store.
> It should answer FAQ/policy questions, check a customer's real order status,
> explain return/refund eligibility, and escalate when it can't resolve
> something — without hallucinating store policy or inventing order data.

## How the weeks build on each other

| Week | Topic | What gets added to the project |
|---|---|---|
| 1–2 | Intro / LLM Applications | `week01_bare_llm/` — a bare LLM chatbot with **no** tools or data. Demonstrates the hallucination problem this whole course exists to solve. |
| 3 | AI Agents | `week03_agent/` — a router that decides FAQ vs. order-lookup vs. return-request and calls a mock order API |
| 4 | RAG | `week04_rag/` — store policy docs get embedded + retrieved so policy answers are grounded |
| 5 | Tokenization | `week05_tokenization/` — tokenize real support queries live, inspect BPE merges |
| 6 | Embeddings | `week06_embeddings/` — visualize FAQ embeddings in 2D |
| 7 | Language Modeling | `week07_language_modeling/` — next-token prediction on a support reply, shown token by token |
| 8 | Transformers | `week08_transformers/` — encoder/decoder walkthrough on a real support query |
| 9 | Attention | `week09_attention/` — attention heatmap on a support query |
| 10 | Training | `week10_training/` — pretraining concepts, toy loss curves |
| 11 | Fine-tuning | `week11_finetuning/` — LoRA/QLoRA fine-tune on support-ticket tone |
| 12 | Prompt Engineering | `week12_prompt_engineering/` — few-shot, CoT, structured JSON output for order data |
| 13 | Agent Frameworks | `week13_agent_frameworks/` — the Week 3 router rebuilt properly in LangGraph |
| 14 | Production | `week14_deployment/` — Docker + AWS EC2 deployment of the full assistant |

Only `week01_bare_llm/` has working code right now — the rest are stubs with a
short README describing what will land there. Ask for any week's code and it
gets built out the same way.

## Shared project data

- `data/orders.json` — mock order records (id, status, items, dates). Not
  wired into the assistant until Week 3 — until then it exists so students can
  see the LLM confidently invent an answer instead of using it.
- `data/policies/*.md` — store policy docs (shipping, returns, warranty) used
  starting Week 4 for RAG.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your API key
```

## Running a given week's demo

```bash
cd week01_bare_llm
streamlit run app.py
```
