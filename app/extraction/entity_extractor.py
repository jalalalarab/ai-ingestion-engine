"""
Entity/relationship extractor — the first step toward a knowledge graph.

One job: given a piece of text (a transcript batch, a document chunk), pull out
TRIPLES — (subject, predicate, object) — the entities and the named relationships
between them. These triples are the atomic unit a knowledge graph is built from:
each triple becomes two nodes (subject, object) joined by an edge (predicate).

We use OpenAI gpt-4o-mini (reusing the same key as vision/Whisper) because
structured extraction is where model quality shows most, and it handles the
mixed Arabic/English transcripts better than the smaller local model. We call it
in JSON mode so the response is guaranteed parseable JSON — no prose, no markdown
fences to scrape.

Config: EXTRACTION_MODEL (default "gpt-4o-mini") and OPENAI_API_KEY from settings.
"""
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)


# The prompt is deliberately strict: return ONLY JSON, one specific shape, and
# normalize entity names so the graph doesn't fragment. The mixed-language note
# is what keeps "Inventory Out" as one node instead of an Arabic and an English
# copy of the same thing.
_EXTRACTION_PROMPT = (
    "You are an information-extraction system building a knowledge graph. "
    "Read the text and extract the key entities and the relationships between them "
    "as triples of the form (subject, predicate, object).\n\n"
    "ENTITY RULES (subject and object):\n"
    "- An entity is a thing: a person, system/module, document, concept, step, "
    "company, field, or decision.\n"
    "- Write EVERY entity in Title Case (e.g. 'Size Order', 'Sales Transaction', "
    "'Payment Terms'). Never use snake_case or lowercase for entities.\n"
    "- The text mixes Arabic and English. Give each entity ONE canonical English "
    "name — prefer the English term the text uses (keep 'Inventory Out', "
    "'Proforma', 'Size Order' in English even inside Arabic sentences). Never "
    "create separate Arabic and English nodes for the same thing.\n"
    "- Use the SAME name every time you refer to the same entity (so it collapses "
    "to one node).\n\n"
    "PREDICATE RULES (the relationship):\n"
    "- Choose the SINGLE best-fit predicate from THIS fixed list, and use it "
    "exactly (lower_snake_case):\n"
    "    is_part_of, becomes, generates, depends_on, has_property, has_field, "
    "affects, does_not_affect, assigned_to, precedes, related_to\n"
    "- Do NOT invent new predicates. If a relationship is a step-to-step flow "
    "(X turns into / converts to / leads to Y), use 'becomes'. If X contains or "
    "belongs to Y, use 'is_part_of'. If nothing fits, use 'related_to'.\n\n"
    "GENERAL RULES:\n"
    "- Only extract relationships actually stated or clearly implied. Do not invent.\n"
    "- Never emit the same relationship twice with different wording.\n"
    "- Skip pleasantries, filler, and greetings.\n\n"
    "Return ONLY a JSON object of this exact shape:\n"
    '{"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}\n'
    "If there are no meaningful relationships, return {\"triples\": []}."
)


def _extract_one(client, text: str) -> list[dict]:
    """Extract triples from a single batch of text. Returns a list of triple dicts."""
    response = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},  # force valid JSON, no prose/fences
        timeout=settings.EXTRACTION_TIMEOUT_SECONDS,
    )
    raw = (response.choices[0].message.content or "") if response.choices else ""
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # JSON mode makes this rare, but never let a bad parse crash the batch.
        logger.warning("Extractor returned unparseable JSON; skipping batch.")
        return []

    triples = data.get("triples", []) if isinstance(data, dict) else []
    # Keep only well-formed triples with all three parts present.
    clean = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        s = _norm_entity(str(t.get("subject", "")))
        p = str(t.get("predicate", "")).strip().lower().replace(" ", "_")
        o = _norm_entity(str(t.get("object", "")))
        if s and p and o:
            clean.append({"subject": s, "predicate": p, "object": o})
    return clean


def _norm_entity(name: str) -> str:
    """
    Safety-net normalization for entity names, in case the model slips on the
    casing rule. Collapses whitespace and Title-cases snake_case/lowercase names
    so 'sales_transaction' and 'Sales Transaction' become one node. Leaves names
    that already contain uppercase (acronyms, proper multi-word) mostly intact.
    """
    name = " ".join(name.replace("_", " ").split())  # collapse ws + de-snake
    if not name:
        return ""
    # If it's all-lowercase, Title-case it. If it already has capitals
    # (e.g. 'TTC Price', 'Inventory Out'), keep as-is to preserve acronyms.
    if name.islower():
        return name.title()
    return name


def _batch_text(segments: list[str], batch_chars: int) -> list[str]:
    """
    Group text segments into batches under batch_chars characters each.

    Same idea as the minutes map-reduce batching: keep each LLM call focused on a
    digestible chunk so extraction doesn't 'skim' a huge input and miss the middle.
    """
    batches: list[str] = []
    current: list[str] = []
    size = 0
    for seg in segments:
        seg = (seg or "").strip()
        if not seg:
            continue
        # +1 for the newline we join with.
        if size + len(seg) + 1 > batch_chars and current:
            batches.append("\n".join(current))
            current = [seg]
            size = len(seg) + 1
        else:
            current.append(seg)
            size += len(seg) + 1
    if current:
        batches.append("\n".join(current))
    return batches


def _dedupe(triples: list[dict]) -> list[dict]:
    """
    Collapse duplicate triples across batches.

    Batches re-extract the same obvious relationships, so without this the same
    fact appears many times. Dedup is case-insensitive on (subject, predicate,
    object), keeping the first-seen casing.
    """
    seen = set()
    out = []
    for t in triples:
        key = (t["subject"].lower(), t["predicate"].lower(), t["object"].lower())
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def extract_triples(segments: list[str]) -> dict:
    """
    Extract deduped triples from a list of text segments (e.g. transcript lines).

    Batches the segments, extracts per batch, merges and dedupes.

    Returns:
        {"triples": [...], "batches_used": N, "raw_count": M, "deduped_count": K}
        so the caller can see how much merging happened.

    Raises:
        RuntimeError: if the OpenAI API key is missing.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — cannot run extraction. Add it to .env."
        )

    from openai import OpenAI  # local import: module still loads if openai absent
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    batches = _batch_text(segments, settings.EXTRACTION_BATCH_CHARS)
    if not batches:
        return {"triples": [], "batches_used": 0, "raw_count": 0, "deduped_count": 0}

    all_triples: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        logger.info("Extracting triples: batch %d/%d (%d chars)", i, len(batches), len(batch))
        all_triples.extend(_extract_one(client, batch))

    deduped = _dedupe(all_triples)
    return {
        "triples": deduped,
        "batches_used": len(batches),
        "raw_count": len(all_triples),
        "deduped_count": len(deduped),
    }
