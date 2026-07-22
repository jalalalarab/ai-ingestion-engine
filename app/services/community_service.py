"""
Community summary service — Phase E: the "long documents".

Two steps:
  1. Detect communities in the graph (GDS Louvain) — the dense clusters of
     related entities.
  2. For each community, write a natural-language summary describing what that
     cluster is about.

GROUNDING: summaries are built from BOTH the cluster's triples AND real source
passages retrieved from Qdrant for those entities. Triples alone were not enough
— a small cluster with one or two thin relationships gave the LLM nothing to
anchor on, so it filled the gap from general knowledge (e.g. describing an ERP
"Import Utility" feature as the economics of importing goods). Feeding it actual
source text removes that vacuum.
"""
import logging

from app.config import settings
from app.embeddings.embedding_client import embed_text
from app.vector_store import graph_store, qdrant_store

logger = logging.getLogger(__name__)


_SUMMARY_PROMPT = (
    "You are summarizing one cluster of a knowledge graph built from ingested "
    "meetings and documents. You are given (a) the cluster's RELATIONSHIPS as "
    "(subject -predicate-> object) facts, and (b) SOURCE PASSAGES from the original "
    "material mentioning these entities.\n\n"
    "Write a concise summary (3-5 sentences) describing what this cluster is about: "
    "the main entities, how they relate, and any process or flow they form.\n\n"
    "CRITICAL: Ground every statement in the relationships and passages provided. "
    "Do NOT use outside/general knowledge, and do NOT guess what a term means from "
    "its name alone — these are domain-specific terms from a specific system, and a "
    "plausible-sounding general definition is usually WRONG. If the provided material "
    "is too thin to describe the cluster meaningfully, say so plainly and just state "
    "which entities it groups together. Write a standalone paragraph, no bullet points."
)


def _grounding_passages(entities: list[str], top_k: int = 4) -> str:
    """
    Retrieve real source passages mentioning this cluster's entities.

    Embeds the entity names as a query and pulls the closest chunks from Qdrant,
    so the summary is anchored in what the source ACTUALLY said about these terms
    rather than what the model assumes they mean.
    """
    if not entities:
        return ""
    query = ", ".join(entities)
    try:
        vec = embed_text(query)
        hits = qdrant_store.search(vec, top_k=top_k)
    except Exception as exc:  # grounding is best-effort; never sink the summary
        logger.warning("Grounding retrieval failed for cluster: %s", exc)
        return ""
    return "\n".join(f"- {h.get('text', '').strip()}" for h in hits if h.get("text"))


def _summarize_community(client, entities: list[str], triples: list[dict]) -> str:
    """Describe one community from its triples plus grounding passages."""
    rels = "\n".join(
        f"({t['subject']}) -{t['predicate']}-> ({t['object']})" for t in triples
    ) or "(no internal relationships)"

    passages = _grounding_passages(entities)

    user_content = (
        f"ENTITIES IN CLUSTER:\n{', '.join(entities)}\n\n"
        f"RELATIONSHIPS:\n{rels}\n\n"
        f"SOURCE PASSAGES:\n{passages or '(none retrieved)'}"
    )

    resp = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": _SUMMARY_PROMPT},
            {"role": "user", "content": user_content},
        ],
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    return (resp.choices[0].message.content or "").strip() if resp.choices else ""


def build_community_summaries(min_size: int = 2) -> dict:
    """
    Detect communities and summarize each into a grounded 'long document'.

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
