"""
Entity resolution — find and merge duplicate nodes in the knowledge graph.

WHY THIS IS DELIBERATELY CONSERVATIVE:
The obvious approach — merge entities whose names are similar (string or embedding
similarity) — is actively dangerous in this domain. The graph contains pairs like
"Size Order" vs "Sales Order", "Size Transaction" vs "Sales Transaction",
"Delivery Out" vs "Sales Invoice": near-identical names that are genuinely
DIFFERENT things. A similarity-based merge would silently collapse them and
corrupt every answer downstream, with no error to notice.

So instead: an LLM judges duplicates (it can reason about meaning, not just
spelling), it is explicitly warned that similar names are often distinct here, and
the default is a DRY RUN — it proposes merges for a human to review before
anything is changed.
"""
import json
import logging

from app.config import settings
from app.vector_store import graph_store

logger = logging.getLogger(__name__)


_RESOLUTION_PROMPT = (
    "You are performing entity resolution on a knowledge graph built from ERP/"
    "business software documentation. You will be given a list of entity names. "
    "Identify ONLY those that refer to the SAME real thing and should be merged "
    "into one node (e.g. an abbreviation and its full form, or a singular/plural "
    "of the identical concept).\n\n"
    "CRITICAL WARNING: In this domain, many entities have SIMILAR NAMES but are "
    "GENUINELY DIFFERENT things. For example 'Size Order' and 'Sales Order' are "
    "different; 'Size Transaction' and 'Sales Transaction' are different. Do NOT "
    "merge two entities just because their names look alike or sound similar. "
    "Merge ONLY when you are confident they denote the identical concept.\n\n"
    "When in doubt, DO NOT merge. A missed merge is harmless; a wrong merge "
    "destroys information.\n\n"
    "Return ONLY JSON of this shape:\n"
    '{"merges": [{"from": "duplicate name", "into": "canonical name", '
    '"reason": "why they are the same"}]}\n'
    "Return {\"merges\": []} if nothing should be merged."
)


def propose_merges() -> list[dict]:
    """Ask the LLM which entity names are duplicates. Returns proposed merges."""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot run entity resolution.")

    names = graph_store.list_entity_names()
    if len(names) < 2:
        return []

    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": _RESOLUTION_PROMPT},
            {"role": "user", "content": "Entity names:\n" + "\n".join(names)},
        ],
        response_format={"type": "json_object"},
        timeout=settings.EXTRACTION_TIMEOUT_SECONDS,
    )
    raw = (resp.choices[0].message.content or "") if resp.choices else ""
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        logger.warning("Entity resolution returned unparseable JSON.")
        return []

    merges = data.get("merges", []) if isinstance(data, dict) else []
    valid = []
    known = set(names)
    for m in merges:
        if not isinstance(m, dict):
            continue
        f, i = str(m.get("from", "")).strip(), str(m.get("into", "")).strip()
        # Only accept merges between names that actually exist in the graph.
        if f and i and f != i and f in known and i in known:
            valid.append({"from": f, "into": i, "reason": str(m.get("reason", "")).strip()})
    return valid


def resolve_entities(apply: bool = False) -> dict:
    """
    Find duplicate entities and (optionally) merge them.

    Args:
        apply: False (default) = DRY RUN, only propose merges. True = actually
               merge them in the graph. Defaulting to dry-run is deliberate: a
               wrong merge is unrecoverable, so a human should see the list first.

    Returns:
        {"applied": bool, "entity_count_before": N, "merges": [...],
         "graph_totals": {...}}
    """
    before = len(graph_store.list_entity_names())
    merges = propose_merges()

    if apply and merges:
        for m in merges:
            logger.info("Merging entity '%s' into '%s'", m["from"], m["into"])
            graph_store.merge_entities(m["from"], m["into"])

    return {
        "applied": bool(apply and merges),
        "entity_count_before": before,
        "merges": merges,
        "graph_totals": graph_store.count_graph(),
    }
