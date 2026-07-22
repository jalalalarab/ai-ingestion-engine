"""
GraphRAG service — Phase D: answer questions using BOTH retrieval paths.

The whole point: plain RAG finds text chunks SIMILAR to the question, which is
great for "what does it say about X" but weak for RELATIONSHIP questions ("what
does Proforma lead to?"), because the answer is a chain spread across the graph,
not a single similar chunk.

GraphRAG runs two paths in parallel and merges them:
  1. VECTOR path  — embed the question, search Qdrant -> similar chunks (prose).
  2. GRAPH path   — LLM extracts the entities the question mentions, resolve them
                    to graph nodes, traverse Neo4j -> relationship chains (triples).
Both go into the LLM's context, so it sees the descriptive text AND the explicit
relationships. This mirrors LightRAG's dual (vector + graph) retrieval, built on
the pipeline we already have rather than adopting a heavy framework.

We keep a plain-RAG mode too, so you can compare answers on the same question and
SEE where the graph actually helps.
"""
import json
import logging

from app.config import settings
from app.embeddings.embedding_client import embed_text, embed_batch
from app.vector_store import qdrant_store, graph_store

logger = logging.getLogger(__name__)


# ---- Step 1: pull the entities a question is about, using the LLM ----

_ENTITY_PROMPT = (
    "Extract the key entities (people, systems, modules, documents, concepts, "
    "steps) mentioned or clearly implied in the user's question. Return canonical "
    "English names in Title Case, matching how they'd appear in a knowledge graph "
    "(e.g. 'Proforma', 'Size Order', 'Sales Transaction'). The question may be in "
    "Arabic or English. Return ONLY JSON: {\"entities\": [\"...\"]}. "
    "If none, return {\"entities\": []}."
)


def _extract_question_entities(question: str) -> list[str]:
    """Ask the LLM which graph entities the question is about. Best-effort."""
    if not settings.OPENAI_API_KEY:
        return []
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=settings.EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": _ENTITY_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            timeout=settings.EXTRACTION_TIMEOUT_SECONDS,
        )
        raw = (resp.choices[0].message.content or "") if resp.choices else ""
        data = json.loads(raw) if raw.strip() else {}
        ents = data.get("entities", []) if isinstance(data, dict) else []
        return [str(e).strip() for e in ents if str(e).strip()]
    except Exception as exc:  # never let entity extraction sink the whole answer
        logger.warning("Question entity extraction failed: %s", exc)
        return []


# ---- Step 2: the two retrieval paths ----

def _vector_context(question: str, top_k: int) -> list[dict]:
    """Vector path: embed the question, return the top_k similar chunks."""
    qvec = embed_text(question)
    return qdrant_store.search(qvec, top_k=top_k)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _rank_triples(question: str, triples: list[dict], keep: int) -> list[dict]:
    """
    Keep only the triples most relevant to the question.

    A 2-hop traversal grabs the whole neighborhood around the anchor entities,
    which on a well-connected node pulls in plenty of facts unrelated to what was
    asked (e.g. every field of Sales Invoice when the question was about Proforma).
    So we embed the question and each triple (as readable text) and keep the top
    `keep` by cosine similarity — the graph half of the context stays focused.

    Best-effort: if embedding fails, fall back to the unranked list rather than
    losing the graph context entirely.
    """
    if len(triples) <= keep:
        return triples
    try:
        texts = [f"{t['subject']} {t['predicate'].replace('_', ' ')} {t['object']}" for t in triples]
        vecs = embed_batch([question] + texts)
        qvec, tvecs = vecs[0], vecs[1:]
        scored = sorted(
            zip(triples, (_cosine(qvec, v) for v in tvecs)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [t for t, _ in scored[:keep]]
    except Exception as exc:
        logger.warning("Triple ranking failed, using unranked triples: %s", exc)
        return triples[:keep]


def _graph_context(question: str, hops: int, max_triples: int = 12) -> dict:
    """Graph path: question -> entities -> resolved nodes -> ranked neighborhood triples."""
    candidates = _extract_question_entities(question)
    resolved = graph_store.find_entities(candidates)
    triples = graph_store.neighborhood_triples(resolved, hops=hops)
    triples = _rank_triples(question, triples, keep=max_triples)
    return {"candidates": candidates, "resolved": resolved, "triples": triples}


# ---- Step 3: build context + answer ----

def _format_chunks(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        # Build a citation label like the agent uses.
        fn = c.get("file_name") or "source"
        if c.get("page_number") is not None:
            label = f"{fn} — p.{c['page_number']}"
        elif c.get("timestamp_seconds") is not None:
            ts = int(c["timestamp_seconds"]); m, s = divmod(ts, 60)
            label = f"{fn} — {m:02d}:{s:02d}"
        else:
            label = fn
        lines.append(f"[{label}] {c.get('text','').strip()}")
    return "\n".join(lines)


def _format_triples(triples: list[dict]) -> str:
    return "\n".join(f"({t['subject']}) -{t['predicate']}-> ({t['object']})" for t in triples)


_ANSWER_PROMPT = (
    "You answer questions using the provided context only. The context has two "
    "parts: PASSAGES (descriptive text with citation labels) and RELATIONSHIPS "
    "(facts from a knowledge graph, as subject -predicate-> object). Use BOTH: the "
    "passages for detail and wording, the relationships to follow chains and answer "
    "'what leads to what' questions. Cite passage labels like (file — 04:32) when "
    "you use them. If the context doesn't answer it, say so. Be concise and grounded."
)


def answer_question(question: str, top_k: int = 5, hops: int = 2, use_graph: bool = True) -> dict:
    """
    Answer a question. With use_graph=True it's GraphRAG (vector + graph); with
    use_graph=False it's plain RAG (vector only) — for side-by-side comparison.

    Returns the answer plus what each path retrieved, so you can SEE the difference.
    """
    chunks = _vector_context(question, top_k)

    graph_info = {"candidates": [], "resolved": [], "triples": []}
    if use_graph:
        graph_info = _graph_context(question, hops)

    passages = _format_chunks(chunks)
    relationships = _format_triples(graph_info["triples"]) if use_graph else ""

    context = f"PASSAGES:\n{passages}"
    if use_graph and relationships:
        context += f"\n\nRELATIONSHIPS:\n{relationships}"

    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,  # gpt-4o-mini: strong, cheap, handles bilingual
        messages=[
            {"role": "system", "content": _ANSWER_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    answer = (resp.choices[0].message.content or "").strip() if resp.choices else ""

    return {
        "question": question,
        "mode": "graphrag" if use_graph else "plain-rag",
        "answer": answer,
        "vector_chunks_used": len(chunks),
        "graph_entities_resolved": graph_info["resolved"],
        "graph_triples_used": graph_info["triples"],
    }
