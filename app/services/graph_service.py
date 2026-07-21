"""
Graph service — builds the knowledge graph for a file.

Ties Phase B (extraction) to Phase C (graph load): extract triples from a file's
transcript, then MERGE them into Neo4j as nodes and relationships. Idempotent —
re-running rebuilds the same file's edges without duplicating them (the store
clears the file first).
"""
import logging

from app.services.extraction_service import extract_from_file
from app.vector_store import graph_store

logger = logging.getLogger(__name__)


def build_graph_for_file(file_id: str) -> dict:
    """
    Extract triples for a file and load them into the graph.

    Steps:
      1. Extract triples from the transcript (Phase B).
      2. Clear this file's existing edges (so a rebuild is clean, not additive).
      3. Load the triples as nodes + relationships.
      4. Return counts, including the whole-graph totals afterward.

    Raises:
        RuntimeError: if the file has no transcript to extract from.
    """
    extraction = extract_from_file(file_id)  # raises RuntimeError if no transcript
    triples = extraction["triples"]
    file_name = extraction.get("file_name")

    # Clear this file's prior edges so re-running doesn't stack duplicates.
    graph_store.clear_file(file_id)

    loaded = graph_store.load_triples(file_id, file_name, triples)
    totals = graph_store.count_graph()

    logger.info(
        "Graph build for '%s': %d triples loaded (graph now %d nodes, %d rels)",
        file_name, loaded, totals["nodes"], totals["relationships"],
    )

    return {
        "file_id": file_id,
        "file_name": file_name,
        "triples_loaded": loaded,
        "graph_totals": totals,
        "extraction": {
            "batches_used": extraction["batches_used"],
            "raw_count": extraction["raw_count"],
            "deduped_count": extraction["deduped_count"],
        },
    }
