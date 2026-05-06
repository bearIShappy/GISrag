"""
rag_pipeline.py  --  GISrag: self-contained RAG pipeline
---------------------------------------------------------
Merges retriever.py + llm.py + pipeline orchestration into one file.

Exports (for rag_query.py backward-compat wrapper):
    embed_query(query)                   -> List[float]
    search_similar(embedding, ...)       -> List[Dict]
    retrieve(query, ...)                 -> List[Dict]
    hybrid_retrieve(query, ...)          -> List[Dict]   # NEW: explicit two-phase flow
    build_context(results, ...)          -> str
    build_prompt(user_query, context)    -> str
    generate(prompt, ...)                -> str
    rag_query(user_query, ...)           -> str
    main()

Stack: PostgreSQL + pgvector + Ollama (gemma3:4b)
No LangChain. No external frameworks.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional

import requests
import psycopg2.extras
from sentence_transformers import SentenceTransformer

from .db_store import get_connection, close_pool, hybrid_search  # ADDED: hybrid_search
from src.config.settings import settings
from src.prompts.prompts import build_prompt as _build_prompt_from_module  # NEW: structured prompt

logger = logging.getLogger(__name__)



# ═══════════════════════════════════════════════════════════════════ #
# 1. EMBEDDING MODEL  (singleton — loaded once, reused across queries)
# ═══════════════════════════════════════════════════════════════════ #

_embed_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer so import is instant."""
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model: %s", settings.text_embedding.MODEL_PATH)
        _embed_model = SentenceTransformer(settings.text_embedding.MODEL_PATH)
    return _embed_model


def embed_query(query: str) -> List[float]:
    """
    Encode a query string into a dense vector.

    mxbai-embed-large-v1 is asymmetric — queries get a prefix,
    documents do not.  The prefix is read from settings so changing
    models only requires updating settings.py.
    """
    model  = _get_model()
    prefix = settings.text_embedding.QUERY_INSTRUCTION   # "Represent this sentence for searching relevant passages: "
    text   = f"{prefix}{query}" if prefix else query
    return model.encode(text, normalize_embeddings=True).tolist()


# ═══════════════════════════════════════════════════════════════════ #
# 2. RETRIEVER  (pgvector cosine-similarity search)
# ═══════════════════════════════════════════════════════════════════ #

def search_similar(
    query_embedding: List[float],
    top_k: int = 5,
    min_score: float = 0.10,
    *,
    place: Optional[str] = None,
    date:  Optional[str] = None,
    role:  Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return the top_k most similar records from PostgreSQL via pgvector.

    Optional metadata filters (place / date / role) are applied as SQL
    WHERE clauses before vector ranking.

    Returns list of dicts with keys:
        id, place, geocoded_place, latitude, longitude,
        date, time, role, summary, source_file, geocode_status,
        similarity (float 0-1)
    """
    emb_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

    filters: list[str] = ["cr.embedding IS NOT NULL"]
    params:  dict[str, Any] = {"emb": emb_literal, "limit": top_k}

    if place:
        filters.append(
            "(cr.place ILIKE %(place_pat)s OR cr.geocoded_place ILIKE %(place_pat)s)"
        )
        params["place_pat"] = f"%{place}%"

    if date:
        filters.append("(cr.date = %(date_val)s OR cr.date LIKE %(date_pre)s)")
        params["date_val"] = date
        params["date_pre"] = f"{date}%"

    if role:
        filters.append("cr.role ILIKE %(role_pat)s")
        params["role_pat"] = f"%{role}%"

    where = " AND ".join(filters)

    sql = f"""
    SELECT
        cr.id,
        cr.place,
        cr.geocoded_place,
        cr.latitude,
        cr.longitude,
        cr.date,
        cr.time,
        cr.role,
        cr.summary,
        cr.source_file,
        cr.geocode_status,
        1 - (cr.embedding <=> %(emb)s::vector) AS similarity
    FROM cleaned_records cr
    WHERE {where}
    ORDER BY cr.embedding <=> %(emb)s::vector
    LIMIT %(limit)s
    """

    t0 = time.perf_counter()
    results: list[dict] = []

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    if row["similarity"] >= min_score:
                        results.append(dict(row))
    except Exception as exc:
        logger.error("pgvector search failed: %s", exc)
        return []

    logger.info(
        "Retriever: %d result(s) in %.3f s  (top_k=%d, min_score=%.2f)",
        len(results), time.perf_counter() - t0, top_k, min_score,
    )
    return results


def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.10,
    *,
    place: Optional[str] = None,
    date:  Optional[str] = None,
    role:  Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience: embed query then search pgvector in one call."""
    if not query or not query.strip():
        logger.warning("retrieve() called with empty query.")
        return []
    return search_similar(
        embed_query(query),
        top_k=top_k, min_score=min_score,
        place=place, date=date, role=role,
    )


# NEW: hybrid_retrieve — explicit two-phase flow
# Phase 1: SQL structured filter  (place / date / category)
# Phase 2: pgvector semantic re-rank on the filtered subset
# Delegates to db_store.hybrid_search() which builds a CTE internally.
def hybrid_retrieve(  # NEW
    query: str,
    top_k: int = 5,
    min_score: float = 0.10,
    *,
    place:    Optional[str] = None,
    date:     Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: SQL filter FIRST, then pgvector re-rank.  # NEW

    Returns the same list[dict] shape as retrieve() / search_similar()
    so it is a drop-in replacement in rag_query().

    Logs both the SQL filter hit count and the final returned count
    so the pipeline step [2/4] shows how much the filter pruned.
    """
    if not query or not query.strip():  # NEW
        logger.warning("hybrid_retrieve() called with empty query.")
        return []

    query_embedding = embed_query(query)  # NEW

    t0 = time.perf_counter()  # NEW
    result = hybrid_search(  # NEW — delegates to db_store.hybrid_search()
        query_embedding,
        place=place,
        date=date,
        category=category,
        top_k=top_k,
        min_score=min_score,
    )
    elapsed = time.perf_counter() - t0  # NEW

    logger.info(  # NEW
        "hybrid_retrieve: SQL filter matched=%d  semantic top_k=%d  returned=%d  (%.3fs)",
        result["filter_count"], top_k, len(result["results"]), elapsed,
    )
    return result["results"]  # NEW — return plain list for build_context()



# ═══════════════════════════════════════════════════════════════════ #
# 3. LLM  (Ollama REST — non-streaming)
# ═══════════════════════════════════════════════════════════════════ #

_OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",    "gemma3:4b")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

_SYSTEM_PROMPT = """You are a GIS-RAG assistant. You answer questions ONLY using the CONTEXT provided below.

Rules:
1. Base your answer ENTIRELY on the context. Do NOT use outside knowledge.
2. If the context does not contain enough information, say: "I don't have enough information in the database to answer that."
3. Cite the source file, place, and date when relevant.
4. Be concise and factual.
5. When locations are mentioned include their coordinates if available.
6. Return your answer as plain text (no JSON, no markdown fences)."""


def build_prompt(user_query: str, context: str) -> str:
    """Assemble the prompt sent to Ollama: system + context + question."""
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{user_query}\n\n"
        f"ANSWER:\n"
    )


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    POST prompt to Ollama /api/generate (non-streaming).
    Raises RuntimeError on connection, timeout, or HTTP errors.
    """
    url = f"{_OLLAMA_URL}/api/generate"
    payload = {
        "model":   model or _OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=_OLLAMA_TIMEOUT)
        resp.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {_OLLAMA_URL}. Run:  ollama serve"
        )
    except requests.Timeout:
        raise RuntimeError(f"Ollama timed out after {_OLLAMA_TIMEOUT}s.")
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Ollama HTTP {exc.response.status_code}: {exc.response.text}"
        )

    body   = resp.json()
    answer = body.get("response", "").strip()

    logger.info(
        "LLM: model=%s  tokens=%s  time=%.2fs",
        payload["model"], body.get("eval_count", "?"), time.perf_counter() - t0,
    )
    return answer or "The model returned an empty response."


# ═══════════════════════════════════════════════════════════════════ #
# 4. CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════ #

_MAX_CONTEXT_CHARS = 4000


def build_context(
    results: List[Dict[str, Any]],
    max_chars: int = _MAX_CONTEXT_CHARS,
) -> str:
    """
    Convert retriever results into a grounded context block for the LLM.

    Each record renders as:
        [Source: X | Place: Y | Date: Z | Role: R | Score: 0.87]
        <summary or structured fallback>

    Truncates once max_chars is reached to stay within the LLM window.
    """
    if not results:
        return "(No relevant records found in the database.)"

    blocks: list[str] = []
    total  = 0

    for r in results:
        place     = r.get("place") or r.get("geocoded_place") or "Unknown"
        date      = r.get("date")        or "N/A"
        source    = r.get("source_file") or "N/A"
        role      = r.get("role")        or "N/A"
        score     = r.get("similarity")
        score_str = f"{score:.2f}" if score is not None else "N/A"

        header  = f"[Source: {source} | Place: {place} | Date: {date} | Role: {role} | Score: {score_str}]"
        content = (r.get("summary") or "").strip()

        if not content:
            parts: list[str] = []
            if place != "Unknown":
                parts.append(f"Location: {place}")
            lat, lon = r.get("latitude"), r.get("longitude")
            if lat is not None and lon is not None:
                parts.append(f"Coordinates: ({lat}, {lon})")
            if date != "N/A":
                parts.append(f"Date: {date}")
            if role != "N/A":
                parts.append(f"Role: {role}")
            content = ". ".join(parts) if parts else "No details available."

        block = f"{header}\n{content}"

        if total + len(block) > max_chars:
            logger.info(
                "Context truncated at %d chars (%d/%d records used).",
                total, len(blocks), len(results),
            )
            break

        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════ #
# 5. RAG ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════ #

def rag_query(
    user_query: str,
    *,
    top_k: int = 5,
    min_score: float = 0.10,
    place:    Optional[str] = None,
    date:     Optional[str] = None,
    role:     Optional[str] = None,
    category: Optional[str] = None,   # NEW: category filter (maps to role column)
    use_hybrid: bool = True,           # NEW: use two-phase hybrid retrieval by default
    max_context_chars: int = _MAX_CONTEXT_CHARS,
) -> str:
    """
    Full RAG pipeline:
        1. Embed query
        2. Retrieve  — hybrid (SQL filter → vector re-rank) OR pure vector  # NEW
        3. Build context
        4. LLM answer

    Optional filters:  place=Leh  date=2024-03  role=Surveyor  category=Flood  top_k=3

    Set use_hybrid=False to fall back to the original pure-vector search_similar().
    """
    if not user_query or not user_query.strip():
        return "Please provide a question."

    pipeline_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("RAG QUERY: %s", user_query)
    if any([place, date, role, category]):
        logger.info(
            "  Filters -> place=%s  date=%s  role=%s  category=%s  hybrid=%s",
            place, date, role, category, use_hybrid,
        )

    # 1 — Embed
    t0 = time.perf_counter()
    query_embedding = embed_query(user_query)
    logger.info("  [1/4] Embedded query  (%.3f s, dim=%d)", time.perf_counter() - t0, len(query_embedding))

    # 2 — Retrieve  (hybrid by default, pure-vector as fallback)  # NEW
    t0 = time.perf_counter()
    if use_hybrid:  # NEW: two-phase hybrid retrieval
        results = hybrid_retrieve(
            user_query,
            top_k=top_k, min_score=min_score,
            place=place, date=date, category=category or role,
        )
    else:  # legacy: pure vector search (backward compatible)
        results = search_similar(
            query_embedding,
            top_k=top_k, min_score=min_score,
            place=place, date=date, role=role,
        )
    logger.info("  [2/4] Retrieved %d result(s)  (%.3f s)", len(results), time.perf_counter() - t0)

    # 3 — Context
    context = build_context(results, max_chars=max_context_chars)
    logger.info("  [3/4] Context built  (%d chars)", len(context))

    if not results:
        return (
            "I could not find any relevant records in the database "
            "for your query. Try rephrasing or broadening the search."
        )

    # 4 — Generate  (uses structured prompt from prompts.py)  # NEW
    t0 = time.perf_counter()
    try:
        # NEW: prefer the structured prompts module; fall back to inline build_prompt
        try:
            prompt = _build_prompt_from_module(user_query, context)
        except Exception:
            prompt = build_prompt(user_query, context)  # backward-compat fallback
        answer = generate(prompt)
    except RuntimeError as exc:
        logger.error("LLM generation failed: %s", exc)
        answer = (
            "LLM is currently unavailable. "
            "Here are the most relevant records:\n\n" + context
        )
    logger.info("  [4/4] LLM answered  (%.2f s)", time.perf_counter() - t0)
    logger.info("  TOTAL: %.2f s", time.perf_counter() - pipeline_start)
    logger.info("=" * 60)

    return answer



# ═══════════════════════════════════════════════════════════════════ #
# 6. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════ #

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("=" * 60)
    print("  GISrag — RAG Pipeline  (pgvector + Ollama)")
    print("=" * 60)

    if len(sys.argv) > 1:
        query  = " ".join(sys.argv[1:])
        answer = rag_query(query)
        print(f"\nQuery: {query}\n{'-'*60}\n{answer}\n{'-'*60}")
        close_pool()
        return

    print("Type your question (or 'quit' to exit).")
    print("Optional inline filters:  place=Leh  date=2024-03  top_k=3\n")

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        # Parse inline key=value filters out of the query string
        kwargs: dict = {}
        clean: list[str] = []
        for tok in raw.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k == "top_k":
                    kwargs["top_k"] = int(v)
                elif k in ("place", "date", "role"):
                    kwargs[k] = v
                else:
                    clean.append(tok)
            else:
                clean.append(tok)

        answer = rag_query(" ".join(clean), **kwargs)
        print(f"{'-'*60}\n{answer}\n{'-'*60}\n")

    close_pool()
    print("\nGoodbye.")


if __name__ == "__main__":
    main()