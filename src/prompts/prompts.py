"""
prompts.py  —  GISrag Prompt Library
-------------------------------------
Centralises all LLM prompt templates for the hybrid RAG pipeline.

Exports:
    SYSTEM_PROMPT   – role declaration + strict rules (used by rag_pipeline)
    build_prompt()  – assembles the final prompt from context + user query
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# Role, rules and output format injected once per LLM call.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a GIS Incident Analysis Assistant specialising in
geospatial event summarisation and structured incident reporting.

You operate in a Hybrid RAG pipeline:
  • STRUCTURED RECORDS  — SQL-filtered rows from PostgreSQL (place, date, role, coords)
  • SEMANTIC MATCHES    — pgvector cosine-similarity results for the user query

STRICT OPERATING RULES:
  1. Answer ONLY from the CONTEXT block provided below.
     Do NOT use any knowledge beyond what is in the context.
  2. If the context does not contain sufficient information to answer,
     respond with exactly:  "Not enough information"
  3. Never fabricate locations, dates, coordinates, or event details.
  4. Never assume or infer missing data — quote only what is explicitly stated.
  5. Prioritise factual precision over answer completeness.
  6. When coordinates are available, always include them in your answer.
  7. Cite the source file and record ID for every key claim.

OUTPUT FORMAT (always use this structure):

Answer:
<A clear, concise, factual 1–3 sentence answer derived strictly from the context.>

Incident Summary:
- Location      : <place or geocoded_place, with coordinates if available>
- Date / Time   : <date and time from the record>
- Event         : <what happened, who was involved (role)>
- Source        : <source_file | Record ID: X | Similarity: Y>

Data Confidence:
<HIGH / MEDIUM / LOW> — state why (e.g. "HIGH: exact place and date match")
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER  # NEW
# Assembles the full prompt sent to the LLM from context + user query.
# Kept separate from rag_pipeline.py so prompt structure can evolve independently.
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(user_query: str, context: str) -> str:  # NEW
    """
    Compose the complete LLM prompt.

    Parameters
    ----------
    user_query : str
        The raw question from the user / API caller.
    context : str
        Pre-built context block from rag_pipeline.build_context().

    Returns
    -------
    str
        Full prompt string ready for the LLM (Ollama / llama.cpp).

    Notes
    -----
    • The CONTEXT section is bounded by XML-style tags so the LLM can
      clearly distinguish retrieved evidence from the instruction text.
    • The QUESTION is placed AFTER the context to reduce positional bias
      (models tend to weight later tokens more strongly).
    """
    return (
        f"{SYSTEM_PROMPT}\n"
        "=" * 60 + "\n"
        "CONTEXT (retrieved records — use ONLY this information):\n"
        "<context>\n"
        f"{context}\n"
        "</context>\n"
        "=" * 60 + "\n"
        f"QUESTION:\n{user_query}\n\n"
        "ANSWER (follow the output format above):\n"
    )