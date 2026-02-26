"""Graph document data structures for nodes, relationships, and documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class Node:
    """A graph node with an id, type label, and properties."""

    id: Union[str, int]
    type: str = "Node"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """A directed relationship between two nodes."""

    source: Node
    target: Node
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDocument:
    """A collection of nodes and relationships extracted from a document."""

    nodes: list[Node] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    source: Any | None = None
