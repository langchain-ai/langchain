"""ArcadeDBGraph — LangChain-compatible graph store backed by ArcadeDB via Bolt.

Drop-in replacement for Neo4jGraph that works with ArcadeDB's native Bolt
endpoint and OpenCypher support. APOC procedures are replaced with pure Cypher.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

from neo4j import GraphDatabase

from langchain_arcadedb.graph_document import GraphDocument, Node
from langchain_arcadedb.schema import format_schema, get_structured_schema

logger = logging.getLogger(__name__)

_DEFAULT_URI = "bolt://localhost:7687"


class ArcadeDBGraph:
    """LangChain GraphStore backed by ArcadeDB over the Bolt protocol.

    Satisfies the ``GraphStore`` protocol expected by ``GraphCypherQAChain``
    and other LangChain graph components.

    Example::

        graph = ArcadeDBGraph(
            url="bolt://localhost:7687",
            username="root",
            password="playwithdata",
            database="mydb",
        )
        graph.refresh_schema()
        print(graph.get_schema)

    Args:
        url: Bolt endpoint URL. Falls back to ``ARCADEDB_URI`` env var.
        username: Database username. Falls back to ``ARCADEDB_USERNAME``.
        password: Database password. Falls back to ``ARCADEDB_PASSWORD``.
        database: Database name. Falls back to ``ARCADEDB_DATABASE``.
        driver_config: Extra keyword arguments passed to the Neo4j driver.
        sample_size: Number of nodes/rels sampled per type during schema
            introspection.
    """

    def __init__(
        self,
        url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        *,
        driver_config: dict[str, Any] | None = None,
        sample_size: int = 25,
    ) -> None:
        self._url = url or os.environ.get("ARCADEDB_URI", _DEFAULT_URI)
        self._username = username or os.environ.get("ARCADEDB_USERNAME", "root")
        self._password = password or os.environ.get(
            "ARCADEDB_PASSWORD", "playwithdata"
        )
        self._database = database or os.environ.get("ARCADEDB_DATABASE", "")
        self._sample_size = sample_size

        config = driver_config or {}
        self._driver = GraphDatabase.driver(
            self._url,
            auth=(self._username, self._password),
            **config,
        )

        self._structured_schema: dict[str, Any] = {}
        self._schema = ""

        try:
            self._driver.verify_connectivity()
        except Exception as e:
            msg = f"Could not connect to ArcadeDB at {self._url}: {e}"
            raise ConnectionError(msg) from e

        self.refresh_schema()

    @property
    def get_schema(self) -> str:
        """Return a human-readable schema string for LLM context."""
        return self._schema

    @property
    def get_structured_schema(self) -> dict[str, Any]:
        """Return the structured schema dictionary."""
        return self._structured_schema

    def query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dicts.

        Args:
            query: Cypher query string.
            params: Query parameters.

        Returns:
            List of result rows as dictionaries.
        """
        params = params or {}
        try:
            if self._database:
                records, _, _ = self._driver.execute_query(
                    query, params, database_=self._database
                )
            else:
                records, _, _ = self._driver.execute_query(query, params)
        except Exception:
            logger.exception("Query failed: %s | params: %s", query, params)
            raise
        return [dict(r) for r in records]

    def refresh_schema(self) -> None:
        """Refresh the cached schema by introspecting the graph."""
        try:
            self._structured_schema = get_structured_schema(
                self._driver, sample_size=self._sample_size
            )
            self._schema = format_schema(self._structured_schema)
        except Exception:
            logger.warning("Schema refresh failed", exc_info=True)
            self._structured_schema = {
                "node_props": {},
                "rel_props": {},
                "relationships": [],
                "metadata": {"constraint": [], "index": []},
            }
            self._schema = ""

    def add_graph_documents(
        self,
        graph_documents: list[GraphDocument],
        *,
        include_source: bool = False,
        base_entity_label: bool = False,
    ) -> None:
        """Import GraphDocument objects into ArcadeDB using pure Cypher.

        Nodes are grouped by type and merged via ``MERGE`` on their ``id``.
        Relationships are grouped by ``(source_type, rel_type, target_type)``
        and merged similarly. This replaces the APOC merge procedures used by
        the Neo4j integration.

        Args:
            graph_documents: Documents to import.
            include_source: If True and the document has a ``source`` with
                ``page_content``, create a ``Document`` node and link entities.
            base_entity_label: Ignored (ArcadeDB uses single type per vertex).
        """
        all_nodes: list[Node] = []
        all_rels: list[dict[str, Any]] = []
        sources: list[tuple[Any, list[Node]]] = []

        for doc in graph_documents:
            all_nodes.extend(doc.nodes)
            for rel in doc.relationships:
                all_rels.append({
                    "source_type": rel.source.type,
                    "source_id": str(rel.source.id),
                    "target_type": rel.target.type,
                    "target_id": str(rel.target.id),
                    "rel_type": rel.type,
                    "properties": rel.properties,
                })
            if include_source and doc.source is not None:
                sources.append((doc.source, doc.nodes))

        # Merge nodes grouped by type
        nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in all_nodes:
            nodes_by_type[node.type].append(
                {"id": str(node.id), "properties": node.properties}
            )

        for node_type, rows in nodes_by_type.items():
            cypher = (
                f"UNWIND $rows AS row "  # noqa: S608
                f"MERGE (n:`{node_type}` {{id: row.id}}) "
                f"SET n += row.properties"
            )
            self.query(cypher, {"rows": rows})

        # Merge relationships grouped by (source_type, rel_type, target_type)
        rels_by_key: dict[
            tuple[str, str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for rel in all_rels:
            key = (rel["source_type"], rel["rel_type"], rel["target_type"])
            rels_by_key[key].append({
                "source_id": rel["source_id"],
                "target_id": rel["target_id"],
                "properties": rel["properties"],
            })

        for (src_type, rel_type, tgt_type), rows in rels_by_key.items():
            cypher = (
                f"UNWIND $rows AS row "  # noqa: S608
                f"MATCH (a:`{src_type}` {{id: row.source_id}}) "
                f"MATCH (b:`{tgt_type}` {{id: row.target_id}}) "
                f"MERGE (a)-[r:`{rel_type}`]->(b) "
                f"SET r += row.properties"
            )
            self.query(cypher, {"rows": rows})

        # Optional: link to source documents
        if include_source and sources:
            for source, nodes in sources:
                page_content = getattr(source, "page_content", str(source))
                self.query(
                    "MERGE (d:Document {content: $content})",
                    {"content": page_content[:5000]},
                )
                for node in nodes:
                    cypher = (
                        f"MATCH (d:Document {{content: $content}}) "  # noqa: S608
                        f"MATCH (n:`{node.type}` {{id: $node_id}}) "
                        f"MERGE (n)-[:MENTIONED_IN]->(d)"
                    )
                    self.query(
                        cypher,
                        {"content": page_content[:5000], "node_id": str(node.id)},
                    )

    def close(self) -> None:
        """Close the Bolt driver connection."""
        self._driver.close()

    def __enter__(self) -> ArcadeDBGraph:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._driver.close()
        except Exception:  # noqa: BLE001, S110
            pass
