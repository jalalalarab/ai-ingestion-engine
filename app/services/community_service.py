"""
Community summary service — Phase E: the "long documents".

Two steps:
  1. Detect communities in the graph (GDS Louvain) — the dense clusters of
     related entities.
  2. For each community, feed its triples to the LLM to write a short
     natural-language summary describing what that cluster is about.

The result is a set of "community summaries" — the long, thematic documents that
let the system answer GLOBAL questions ("what are the main themes?") that local
graph traversal or vector search handle poorly. This is what a GraphRAG framework
generates for you; here we build it on our own graph so it's fully explainable.
"""
import logging

from app.config import settings
from app.vector_store import graph_store

logger = logging.getLogger(__name__)


_SUMMARY_PROMPT = (
    "You are summarizing one cluster of a knowledge graph built from a meeting/"
    "document. You are given the cluster's relationships as (subject -predicate-> "
    "object) facts. Write a concise natural-language summary (3-5 sentences) that "
    "describes what this cluster is about: the main entities, how they relate, and "
    "any process or flow they form. Write it as a standalone paragraph someone "
    "could read to understand this theme without seeing the graph. Do not invent "
    "facts beyond the relationships given. Do not use bullet points."
)


def _summarize_community(client, entities: list[str], triples: list[dict]) -> str:
    """Ask the LLM to describe one community from its triples."""
    if not triples:
        # A community with no internal edges — just name its entities.
        return "This cluster contains: " + ", ".join(entities) + "."

    rels = "\n".join(
        f"({t['subject']}) -{t['predicate']}-> ({t['object']})" for t in triples
    )
    resp = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": _SUMMARY_PROMPT},
            {"role": "user", "content": f"Cluster relationships:\n{rels}"},
        ],
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    return (resp.choices[0].message.content or "").strip() if resp.choices else ""


def build_community_summaries(min_size: int = 2) -> dict:
    """
    Detect communities and summarize each into a 'long document'.

    Args:
        min_size: skip trivially-small communities (fewer than this many
                  entities) — a lone node isn't a theme worth summarizing.

    Returns:
        {"community_count": N, "summaries": [
            {"community": id, "entities": [...], "size": k, "summary": "..."}
        ]}
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot summarize communities.")

    communities = graph_store.detect_communities()
    communities = [c for c in communities if len(c["entities"]) >= min_size]

    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    summaries = []
    for c in communities:
        logger.info("Summarizing community %s (%d entities)", c["community"], len(c["entities"]))
        text = _summarize_community(client, c["entities"], c["triples"])
        summaries.append({
            "community": c["community"],
            "entities": c["entities"],
            "size": len(c["entities"]),
            "summary": text,
        })

    return {"community_count": len(summaries), "summaries": summaries}
