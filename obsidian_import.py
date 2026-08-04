"""
Obsidian import — the reverse of obsidian_export (closes the loop).

Reads an edited Obsidian vault (a folder of Markdown notes) and turns it back
into triples, so a human's corrections to the notes become the graph.

The flow this completes:
    video -> AI generates notes -> HUMAN EDITS notes -> [this] -> graph updated
             (obsidian_export)                          (obsidian_import)

So if extraction got a relationship wrong, someone fixes the .md note, runs this,
and the knowledge graph reflects the correction. The notes become the editable
source of truth.

How it parses (the exact reverse of the export):
  A note file "Proforma.md" whose ## Relationships section contains
      - generates [[Size Order]]
  becomes the triple (Proforma, generates, Size Order).
  - The FILENAME (minus .md) is the subject.
  - Each [[link]] under ## Relationships is an object.
  - The words between the "- " and the "[[" are the predicate (re-snaked).

Only the ## Relationships section is read. The ## Referenced by section is the
mirror of other notes' outgoing links, so reading it too would double-count.

Applying to the graph: this rebuilds a file's edges from the notes (clear + load
via graph_store), so edits AND deletions both take effect — deleting a
relationship line in a note removes it from the graph on re-import.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches a relationship line like "- generates [[Size Order]]" and captures
# the predicate phrase ("generates") and the linked entity ("Size Order").
_REL_LINE = re.compile(r"^\s*-\s+(.*?)\s*\[\[([^\]]+)\]\]\s*$")


def _phrase_to_predicate(phrase: str) -> str:
    """Reverse of the export's readable form: 'is part of' -> 'is_part_of'."""
    p = phrase.strip().lower()
    return re.sub(r"\s+", "_", p) if p else "related_to"


def parse_vault(vault_dir: str) -> list[dict]:
    """
    Parse an Obsidian vault folder into triples.

    Reads every .md file, takes the filename as the subject, and each
    "- <predicate> [[Object]]" line under the "## Relationships" heading as a
    (subject, predicate, object) triple.

    Returns:
        A list of {"subject", "predicate", "object"} dicts. Empty if no notes or
        no relationships found.
    """
    vault = Path(vault_dir)
    if not vault.is_dir():
        raise RuntimeError(f"Vault folder not found: {vault_dir}")

    triples: list[dict] = []

    for note in sorted(vault.glob("*.md")):
        subject = note.stem  # filename without .md
        lines = note.read_text(encoding="utf-8").splitlines()

        in_relationships = False
        for line in lines:
            stripped = line.strip()

            # Track which section we're in. Only parse under "## Relationships".
            if stripped.startswith("## "):
                in_relationships = stripped.lower() == "## relationships"
                continue

            if not in_relationships:
                continue

            m = _REL_LINE.match(line)
            if m:
                phrase, obj = m.group(1), m.group(2).strip()
                if obj:
                    triples.append({
                        "subject": subject,
                        "predicate": _phrase_to_predicate(phrase),
                        "object": obj,
                    })

    logger.info("Parsed %d triples from vault '%s'", len(triples), vault_dir)
    return triples


def import_vault_to_graph(vault_dir: str, file_id: str, file_name: str | None = None) -> dict:
    """
    Parse an edited vault and reload it into Neo4j as the graph for `file_id`.

    This REPLACES that file's edges with what the notes now say (clear + load via
    graph_store), so human edits and deletions in the notes take effect.

    Args:
        vault_dir: the Obsidian vault folder to read.
        file_id:   the file whose graph these notes represent (its edges are
                   cleared and rebuilt from the notes).
        file_name: optional display name stored on the relationships.

    Returns:
        {"triples_parsed": int, "graph_totals": {...}}
    """
    # Import here so parse_vault stays usable/testable without a live Neo4j.
    from app.vector_store.graph_store import load_triples, clear_file, count_graph

    triples = parse_vault(vault_dir)

    # Clear this file's existing edges, then load the (edited) triples fresh.
    clear_file(file_id)
    if triples:
        load_triples(file_id=file_id, file_name=file_name, triples=triples)

    totals = count_graph()
    logger.info(
        "Imported %d triples from vault into graph for file_id=%s",
        len(triples), file_id[:8] if file_id else "?",
    )
    return {"triples_parsed": len(triples), "graph_totals": totals}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("Usage: python obsidian_import.py <vault_folder> <file_id> [file_name]")
        print("Example: python obsidian_import.py noria_vault b57db252-... sales_noria_erp.mp4")
        print()
        print("Or to just preview parsed triples without touching the graph:")
        print("  python obsidian_import.py <vault_folder> --preview")
        sys.exit(1)

    if sys.argv[2] == "--preview":
        parsed = parse_vault(sys.argv[1])
        print(f"Parsed {len(parsed)} triples (preview — graph NOT modified):\n")
        for t in parsed[:40]:
            print(f"  ({t['subject']}) -{t['predicate']}-> ({t['object']})")
        if len(parsed) > 40:
            print(f"  ... and {len(parsed) - 40} more")
    else:
        fid = sys.argv[2]
        fname = sys.argv[3] if len(sys.argv) > 3 else None
        result = import_vault_to_graph(sys.argv[1], fid, fname)
        print(f"Imported {result['triples_parsed']} triples from the vault.")
        print(f"Graph now has {result['graph_totals']}.")
