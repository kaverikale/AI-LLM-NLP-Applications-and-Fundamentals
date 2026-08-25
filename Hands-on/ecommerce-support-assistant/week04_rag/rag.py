"""
rag.py — the Retrieval half of RAG: chunking, embeddings, search.

This is the file Lecture 4 is about. It never talks to Claude — it's
a small local "vector database" built once at import time (same
pattern as tools.py's module-level ORDERS dict), searched with cosine
similarity.

Pipeline (see Lecture 4, Part 6's "complete RAG pipeline" diagram):
    documents.POLICY_DOCUMENTS -> chunk -> embed -> in-memory index   (indexing, once)
    user query -> embed -> cosine similarity -> top-k chunks         (retrieval, per question)

Swap points for later, without touching planner.py or tools.py:
  - chunk_documents(): paragraph-based here; swap for recursive/
    sentence-based splitting once these are real PDFs instead of
    short mock strings.
  - _MODEL: a local sentence-transformers model, so this demo needs
    no second API key. Swap for Voyage AI or OpenAI embeddings by
    replacing embed_texts() alone — everything downstream (build_index,
    search) only cares that it gets back a vector.
  - the in-memory list _INDEX: swap for a real vector DB (Chroma,
    Pinecone, pgvector...) by replacing build_index()/search() while
    keeping their signatures the same.

Requires: pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from documents import POLICY_DOCUMENTS

TOP_K = 3

# Small, fast, local embedding model — good enough for a classroom
# demo and needs no API key. First run downloads ~80MB; after that
# it's cached and loads instantly.
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Paragraph-based chunking (Lecture 4, Part 5): split each
    document's text on blank lines. This is enough here because each
    mock policy doc is already written as short, self-contained
    paragraphs — one topic per paragraph, no overlap needed."""
    chunks = []
    for doc in documents:
        paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            chunks.append({
                "text": paragraph,
                "source": doc["source"],
                "chunk_index": i,
            })
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """Text -> vector, batched. Normalized so cosine similarity
    reduces to a plain dot product in search() below."""
    return _MODEL.encode(texts, normalize_embeddings=True)


def build_index(documents: list[dict]) -> list[dict]:
    """The indexing phase: chunk every document, embed every chunk
    once, and keep vector + text + metadata together. This in-memory
    list *is* our vector database for the demo."""
    print(f"in build_index: indexing {len(documents)} document(s)...")
    chunks = chunk_documents(documents)
    vectors = embed_texts([c["text"] for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    print(f"in build_index: indexed {len(chunks)} chunk(s)")
    return chunks


# Built once, at import time — mirrors tools.py's module-level ORDERS.
_INDEX = build_index(POLICY_DOCUMENTS)


def search(query: str, k: int = TOP_K) -> list[dict]:
    """The retrieval phase: embed the query, score it against every
    indexed chunk with cosine similarity, return the top-k. Since
    everything is normalized, cosine similarity is just a dot
    product."""
    print("in search:", query)
    query_vector = embed_texts([query])[0]
    scored = [
        {**chunk, "score": float(np.dot(query_vector, chunk["embedding"]))}
        for chunk in _INDEX
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]


def search_policies(query: str) -> dict:
    """Tool-facing wrapper: same shape as tools.py's other tool
    functions (plain args in, JSON-serializable dict out). Registered
    in TOOLS_SCHEMA/TOOL_FUNCTIONS so the planner calls this exactly
    like get_order_status or check_return_eligibility — see tools.py."""
    results = search(query)
    return {
        "results": [
            {"source": r["source"], "excerpt": r["text"], "relevance": round(r["score"], 3)}
            for r in results
        ]
    }