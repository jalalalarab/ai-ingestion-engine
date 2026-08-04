"""
Obsidian export — Step 6 of the video-synthesis pipeline.

Turns the Neo4j knowledge graph into an OBSIDIAN VAULT: a folder of Markdown
notes, one per entity, where each relationship is a [[wiki-link]] to another
note. Open the folder in Obsidian and it draws the knowledge graph — but every
node is now a human-readable, editable text file.

Why this exists (the instructor's direction):
  Neo4j is a machine's graph — fast to query, but a human can't open it and fix
  a wrong relationship. Obsidian is a human's graph — the same nodes and edges,
  but stored as editable Markdown. This is the "human-editable layer above the
  graph": if extraction got something wrong, an EDM employee edits the .md note,
  and the graph can be regenerated from the corrected notes.

  The vault is generated FROM the long document's graph (per the decision that
  the Obsidian graph is built from the synthesized document, not hand-written).

What one note looks like:
    # Proforma

    ## Relationships
    - generates [[Size Order]]
    - related to [[Inventory Control System]]

    ## Mentioned in
    - sales_noria_erp.mp4

Opening the vault in Obsidian's graph view renders Proforma linked to Size Order
and Inventory Control System — the knowledge graph, human-editable.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from app.vector_store.graph_store import get_driver

logger = logging.getLogger(__name__)


def get_all_triples() -> list[dict]:
    """
    Read the ENTIRE knowledge graph from Neo4j as a flat list of triples.

    Returns:
        A list of {"subject", "predicate", "object", "file_name"} dicts — one per
        relationship in the graph. Empty list if the graph is empty.
    """
    driver = get_driver()
    triples: list[dict] = []
    with driver.session() as session:
        result = session.run(
            "MATCH (s:Entity)-[r:REL]->(o:Entity) "
            "RETURN s.name AS subject, r.predicate AS predicate, "
            "o.name AS object, r.file_name AS file_name"
        )
        for record in result:
            triples.append({
                "subject": record["subject"],
                "predicate": record["predicate"],
                "object": record["object"],
                "file_name": record.get("file_name"),
            })
    logger.info("Read %d triples from the graph for Obsidian export", len(triples))
    return triples


def _safe_filename(name: str) -> str:
    """
    Make an entity name safe as a filename. Obsidian links use the note's
    basename, so the filename must match the [[link]] text — we keep the name
    as-is except for characters illegal in filenames.
    """
    # Remove characters that are illegal in Windows/Unix filenames.
    cleaned = re.sub(r'[<>:"/\\|?*]', "", name).strip()
    return cleaned or "Unnamed"


def _predicate_to_phrase(predicate: str) -> str:
    """Turn a snake_case predicate into readable link text: is_part_of -> 'is part of'."""
    return predicate.replace("_", " ")


def export_to_obsidian(output_dir: str) -> dict:
    """
    Generate an Obsidian vault (folder of linked Markdown notes) from the graph.

    One note per entity. Outgoing relationships become [[wiki-links]] under a
    "Relationships" heading; incoming relationships are noted too so the graph is
    fully navigable from either end. Source files are listed under "Mentioned in".

    Args:
        output_dir: folder to write the vault into (created if missing).

    Returns:
        {"entities": int, "relationships": int, "notes_written": int, "path": str}
    """
    triples = get_all_triples()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not triples:
        logger.warning("Graph is empty — no Obsidian notes to write.")
        return {"entities": 0, "relationships": 0, "notes_written": 0, "path": str(out)}

    # Gather, per entity: outgoing links, incoming links, and source files.
    # Using dicts keyed by entity name so every entity gets exactly one note.
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    sources: dict[str, set[str]] = {}

    def _ensure(entity: str) -> None:
        outgoing.setdefault(entity, [])
        incoming.setdefault(entity, [])
        sources.setdefault(entity, set())

    for t in triples:
        s, p, o = t["subject"], t["predicate"], t["object"]
        _ensure(s)
        _ensure(o)
        phrase = _predicate_to_phrase(p)
        outgoing[s].append(f"- {phrase} [[{_safe_filename(o)}]]")
        incoming[o].append(f"- [[{_safe_filename(s)}]] {phrase} this")
        if t.get("file_name"):
            sources[s].add(t["file_name"])
            sources[o].add(t["file_name"])

    notes_written = 0
    for entity in outgoing:  # every entity is a key here
        lines = [f"# {entity}", ""]

        if outgoing[entity]:
            lines.append("## Relationships")
            lines.extend(sorted(set(outgoing[entity])))
            lines.append("")

        if incoming[entity]:
            lines.append("## Referenced by")
            lines.extend(sorted(set(incoming[entity])))
            lines.append("")

        if sources[entity]:
            lines.append("## Mentioned in")
            lines.extend(f"- {src}" for src in sorted(sources[entity]))
            lines.append("")

        note_path = out / f"{_safe_filename(entity)}.md"
        note_path.write_text("\n".join(lines), encoding="utf-8")
        notes_written += 1

    logger.info(
        "Obsidian export: %d entities, %d relationships -> %d notes in %s",
        len(outgoing), len(triples), notes_written, out,
    )
    return {
        "entities": len(outgoing),
        "relationships": len(triples),
        "notes_written": notes_written,
        "path": str(out),
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else "obsidian_vault"
    result = export_to_obsidian(target)
    print(f"Wrote {result['notes_written']} notes "
          f"({result['entities']} entities, {result['relationships']} relationships) "
          f"to: {result['path']}")
    print("Open that folder in Obsidian (File -> Open Vault) to see the graph.")
