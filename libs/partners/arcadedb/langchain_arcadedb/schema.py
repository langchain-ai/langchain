"""Schema introspection for ArcadeDB via Bolt protocol.

Replaces neo4j_graphrag schema helpers and APOC meta procedures with
pure Cypher queries compatible with ArcadeDB's OpenCypher implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver

logger = logging.getLogger(__name__)

# Map Python types to schema type strings (matches neo4j_graphrag format).
_TYPE_MAP: dict[type, str] = {
    int: "INTEGER",
    float: "FLOAT",
    str: "STRING",
    bool: "BOOLEAN",
    list: "LIST",
}


def _infer_type(value: Any) -> str:
    """Infer a schema type string from a Python value."""
    for py_type, type_str in _TYPE_MAP.items():
        if isinstance(value, py_type):
            return type_str
    return "STRING"


def _get_node_properties(
    driver: Driver, label: str, sample_size: int = 25
) -> list[dict[str, str]]:
    """Sample nodes of a given label and infer property names and types."""
    query = f"MATCH (n:`{label}`) RETURN n LIMIT $limit"  # noqa: S608
    records, _, _ = driver.execute_query(query, {"limit": sample_size})

    props: dict[str, str] = {}
    for record in records:
        node = record["n"]
        for key, value in dict(node).items():
            if key not in props and value is not None:
                props[key] = _infer_type(value)

    return [{"property": k, "type": v} for k, v in props.items()]


def _get_rel_properties(
    driver: Driver, rel_type: str, sample_size: int = 25
) -> list[dict[str, str]]:
    """Sample relationships of a given type and infer property names and types."""
    query = f"MATCH ()-[r:`{rel_type}`]->() RETURN r LIMIT $limit"  # noqa: S608
    records, _, _ = driver.execute_query(query, {"limit": sample_size})

    props: dict[str, str] = {}
    for record in records:
        rel = record["r"]
        for key, value in dict(rel).items():
            if key not in props and value is not None:
                props[key] = _infer_type(value)

    return [{"property": k, "type": v} for k, v in props.items()]


def _get_node_labels(driver: Driver) -> list[str]:
    """Get all node labels in the graph."""
    query = "MATCH (n) RETURN DISTINCT labels(n)[0] AS label"
    records, _, _ = driver.execute_query(query)
    return [r["label"] for r in records if r["label"] is not None]


def _get_relationship_types(driver: Driver) -> list[str]:
    """Get all relationship types in the graph."""
    query = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS type"
    records, _, _ = driver.execute_query(query)
    return [r["type"] for r in records if r["type"] is not None]


def _get_relationship_patterns(driver: Driver) -> list[dict[str, str]]:
    """Get distinct (source_label, rel_type, target_label) triples."""
    query = (
        "MATCH (a)-[r]->(b) "
        "RETURN DISTINCT labels(a)[0] AS start, type(r) AS type, labels(b)[0] AS end"
    )
    records, _, _ = driver.execute_query(query)
    return [
        {"start": r["start"], "type": r["type"], "end": r["end"]}
        for r in records
        if r["start"] is not None and r["end"] is not None
    ]


def get_structured_schema(
    driver: Driver,
    sample_size: int = 25,
) -> dict[str, Any]:
    """Build a structured schema dictionary by introspecting the ArcadeDB graph.

    Returns a dict matching the neo4j_graphrag format::

        {
            "node_props": {"Person": [{"property": "name", "type": "STRING"}]},
            "rel_props": {"KNOWS": [...]},
            "relationships": [{"start": "Person", "type": "KNOWS", "end": "Person"}],
            "metadata": {"constraint": [], "index": []}
        }

    Args:
        driver: A Neo4j Python driver connected to ArcadeDB via Bolt.
        sample_size: Number of nodes/rels to sample per type for type inference.

    Returns:
        Structured schema dictionary.
    """
    labels = _get_node_labels(driver)
    rel_types = _get_relationship_types(driver)

    node_props: dict[str, list[dict[str, str]]] = {}
    for label in labels:
        try:
            node_props[label] = _get_node_properties(driver, label, sample_size)
        except Exception:
            logger.warning("Failed to get properties for label %s", label)

    rel_props: dict[str, list[dict[str, str]]] = {}
    for rel_type in rel_types:
        try:
            rel_props[rel_type] = _get_rel_properties(
                driver, rel_type, sample_size
            )
        except Exception:
            logger.warning("Failed to get properties for rel type %s", rel_type)

    relationships = _get_relationship_patterns(driver)

    return {
        "node_props": node_props,
        "rel_props": rel_props,
        "relationships": relationships,
        "metadata": {"constraint": [], "index": []},
    }


def format_schema(structured_schema: dict[str, Any]) -> str:
    """Format a structured schema dict into a human-readable string.

    This string is used as context for LLM Cypher generation.

    Args:
        structured_schema: Output of ``get_structured_schema()``.

    Returns:
        Multi-line string describing node/rel properties and patterns.
    """
    lines: list[str] = []

    lines.append("Node properties:")
    for label, props in structured_schema.get("node_props", {}).items():
        prop_strs = [f"{p['property']}: {p['type']}" for p in props]
        lines.append(f"  {label} {{{', '.join(prop_strs)}}}")

    lines.append("Relationship properties:")
    for rel_type, props in structured_schema.get("rel_props", {}).items():
        prop_strs = [f"{p['property']}: {p['type']}" for p in props]
        lines.append(f"  {rel_type} {{{', '.join(prop_strs)}}}")

    lines.append("Relationships:")
    for rel in structured_schema.get("relationships", []):
        lines.append(f"  (:{rel['start']})-[:{rel['type']}]->(:{rel['end']})")

    return "\n".join(lines)
