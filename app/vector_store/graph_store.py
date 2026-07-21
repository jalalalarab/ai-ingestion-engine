"""
Neo4j graph store — the only module that talks to Neo4j.

Mirrors qdrant_store's design: one gateway module, a shared lazy driver, and a
handful of focused functions. Where qdrant_store stores vectors, this stores the
knowledge graph — the entities and relationships extracted in Phase B become
nodes and edges here.

Provides:
- get_driver(): shared Neo4j driver, created on first use.
- load_triples(): MERGE a batch of (subject, predicate, object) triples as
  nodes + relationships, tagged with the file they came from.
- count_graph(): node and relationship counts (diagnostic).
- clear_file(): remove everything ingested from one file.
- clear_all(): wipe the whole graph (dev convenience).
"""
import logging

from neo4j import GraphDatabase

from app.config import settings

logger = logging.getLogger(__name__)

_driver = None


def get_driver():
    """Return a shared Neo4j driver, creating it on first call."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


def verify_connectivity() -> None:
    """Raise if Neo4j isn't reachable — a clear failure beats a confusing one later."""
    get_driver().verify_connectivity()


def load_triples(file_id: str, file_name: str | None, triples: list[dict]) -> int:
    """
    Load triples into the graph as nodes and relationships.

    Each entity becomes an :Entity node keyed by its name (MERGE = create if new,
    reuse if it already exists — so the same entity across meetings collapses to
    one node). Each triple becomes a :REL relationship carrying its predicate as a
    property, tagged with the file_id it came from (so we can trace provenance and
    delete per-file later).

    Why store the predicate as a PROPERTY rather than as the relationship TYPE:
    relationship types can't be parameterized safely in Cypher (they'd require
    string-building, which risks injection and makes MERGE awkward). Storing
    predicate as a property keeps the load simple and safe; we can still query by
    it. A later refinement could promote common predicates to real types.

    Returns the number of triples loaded.
    """
    if not triples:
        return 0

    # One parameterized query, run per triple. MERGE nodes by name so repeats
    # dedupe into single nodes; MERGE the relationship by (predicate, file_id) so
    # re-running the same file doesn't pile up duplicate edges.
    cypher = """
    MERGE (s:Entity {name: $subject})
    MERGE (o:Entity {name: $object})
    MERGE (s)-[r:REL {predicate: $predicate, file_id: $file_id}]->(o)
    SET r.file_name = $file_name
    """

    driver = get_driver()
    loaded = 0
    with driver.session() as session:
        for t in triples:
            session.run(
                cypher,
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                file_id=file_id,
                file_name=file_name,
            )
            loaded += 1
    return loaded


def count_graph() -> dict:
    """Return {'nodes': N, 'relationships': M} for the whole graph."""
    driver = get_driver()
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return {"nodes": nodes, "relationships": rels}


def clear_file(file_id: str) -> None:
    """
    Remove relationships (and now-orphaned nodes) that came from one file.

    Deletes REL edges tagged with this file_id, then removes any Entity node left
    with no relationships at all (so wiping a file doesn't leave dangling nodes).
    """
    driver = get_driver()
    with driver.session() as session:
        session.run("MATCH ()-[r:REL {file_id: $fid}]->() DELETE r", fid=file_id)
        session.run("MATCH (n:Entity) WHERE NOT (n)--() DELETE n")


def clear_all() -> None:
    """Wipe the entire graph. Dev convenience — use with care."""
    driver = get_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def find_entities(names: list[str]) -> list[str]:
    """
    Resolve a list of candidate entity names to actual node names in the graph.

    Case-insensitive and fuzzy: a candidate matches a node if either contains the
    other (so "proforma" finds "Proforma", and "size order" finds "Size Order").
    Returns the DISTINCT real node names that matched — these are the anchors we
    traverse from.
    """
    if not names:
        return []

    cypher = """
    UNWIND $names AS candidate
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower(candidate)
       OR toLower(candidate) CONTAINS toLower(e.name)
    RETURN DISTINCT e.name AS name
    """
    driver = get_driver()
    with driver.session() as session:
        rows = session.run(cypher, names=names)
        return [r["name"] for r in rows]


def neighborhood_triples(entity_names: list[str], hops: int = 2, limit: int = 60) -> list[dict]:
    """
    Traverse the graph outward from the given entities and return the triples found.

    Walks up to `hops` relationships away from each anchor entity (both directions),
    collecting the (subject, predicate, object) triples along those paths. This is
    the graph half of GraphRAG: instead of similar text, it returns the RELATIONSHIP
    CHAINS around the entities a question mentions — e.g. from "Proforma" it pulls
    Proforma→Size Order→Delivery Out→Sales Transaction.

    Args:
        entity_names: resolved node names (from find_entities).
        hops: how many relationships to traverse out (2 catches most useful chains).
        limit: safety cap on triples returned, so a hub node can't flood context.

    Returns:
        A list of {subject, predicate, object} dicts, de-duplicated.
    """
    if not entity_names:
        return []

    # Variable-length path up to `hops`, undirected traversal so we catch chains
    # that point toward the anchor as well as away from it. We return each edge's
    # endpoints and predicate. `hops` is inlined (validated int) because Cypher
    # can't parameterize the path-length bound.
    hops = max(1, min(int(hops), 4))  # clamp to a sane range
    cypher = f"""
    MATCH (a:Entity)
    WHERE a.name IN $names
    MATCH path = (a)-[:REL*1..{hops}]-(b:Entity)
    WITH relationships(path) AS rels
    UNWIND rels AS r
    WITH startNode(r) AS s, r AS r, endNode(r) AS o
    RETURN DISTINCT s.name AS subject, r.predicate AS predicate, o.name AS object
    LIMIT $limit
    """
    driver = get_driver()
    with driver.session() as session:
        rows = session.run(cypher, names=entity_names, limit=limit)
        return [
            {"subject": r["subject"], "predicate": r["predicate"], "object": r["object"]}
            for r in rows
        ]
