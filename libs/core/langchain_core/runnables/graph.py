from __future__ import annotations

import inspect
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypedDict,
    Union,
    overload,
)
from uuid import UUID, uuid4

from langchain_core.pydantic_v1 import BaseModel

if TYPE_CHECKING:
    from langchain_core.runnables.base import Runnable as RunnableType


class Stringifiable(Protocol):
    def __str__(self) -> str: ...


class LabelsDict(TypedDict):
    """Dictionary of labels for nodes and edges in a graph."""

    nodes: dict[str, str]
    """Labels for nodes."""
    edges: dict[str, str]
    """Labels for edges."""


def is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID.

    Args:
        value: The string to check.

    Returns:
        True if the string is a valid UUID, False otherwise.
    """
    try:
        UUID(value)
        return True
    except ValueError:
        return False


class Edge(NamedTuple):
    """Edge in a graph."""

    source: str
    target: str
    data: Optional[Stringifiable] = None
    conditional: bool = False

    def copy(
        self, *, source: Optional[str] = None, target: Optional[str] = None
    ) -> Edge:
        return Edge(
            source=source or self.source,
            target=target or self.target,
            data=self.data,
            conditional=self.conditional,
        )


class Node(NamedTuple):
    """Node in a graph."""

    id: str
    name: str
    data: Union[Type[BaseModel], RunnableType]
    metadata: Optional[Dict[str, Any]]

    def copy(self, *, id: Optional[str] = None, name: Optional[str] = None) -> Node:
        return Node(
            id=id or self.id,
            name=name or self.name,
            data=self.data,
            metadata=self.metadata,
        )


class Branch(NamedTuple):
    """Branch in a graph."""

    condition: Callable[..., str]
    ends: Optional[dict[str, str]]


class CurveStyle(Enum):
    """Enum for different curve styles supported by Mermaid"""

    BASIS = "basis"
    BUMP_X = "bumpX"
    BUMP_Y = "bumpY"
    CARDINAL = "cardinal"
    CATMULL_ROM = "catmullRom"
    LINEAR = "linear"
    MONOTONE_X = "monotoneX"
    MONOTONE_Y = "monotoneY"
    NATURAL = "natural"
    STEP = "step"
    STEP_AFTER = "stepAfter"
    STEP_BEFORE = "stepBefore"


@dataclass
class NodeStyles:
    """Schema for Hexadecimal color codes for different node types"""

    default: str = "fill:#f2f0ff,line-height:1.2"
    first: str = "fill-opacity:0"
    last: str = "fill:#bfb6fc"


class MermaidDrawMethod(Enum):
    """Enum for different draw methods supported by Mermaid"""

    PYPPETEER = "pyppeteer"  # Uses Pyppeteer to render the graph
    API = "api"  # Uses Mermaid.INK API to render the graph


def node_data_str(id: str, data: Union[Type[BaseModel], RunnableType]) -> str:
    """Convert the data of a node to a string.

    Args:
        id: The node id.
        data: The node data.

    Returns:
        A string representation of the data.
    """
    from langchain_core.runnables.base import Runnable

    if not is_uuid(id):
        return id
    elif isinstance(data, Runnable):
        data_str = data.get_name()
    else:
        data_str = data.__name__
    return data_str if not data_str.startswith("Runnable") else data_str[8:]


def node_data_json(
    node: Node, *, with_schemas: bool = False
) -> Dict[str, Union[str, Dict[str, Any]]]:
    """Convert the data of a node to a JSON-serializable format.

    Args:
        node: The node to convert.
        with_schemas: Whether to include the schema of the data if
            it is a Pydantic model.

    Returns:
        A dictionary with the type of the data and the data itself.
    """
    from langchain_core.load.serializable import to_json_not_implemented
    from langchain_core.runnables.base import Runnable, RunnableSerializable

    if isinstance(node.data, RunnableSerializable):
        json: Dict[str, Any] = {
            "type": "runnable",
            "data": {
                "id": node.data.lc_id(),
                "name": node_data_str(node.id, node.data),
            },
        }
    elif isinstance(node.data, Runnable):
        json = {
            "type": "runnable",
            "data": {
                "id": to_json_not_implemented(node.data)["id"],
                "name": node_data_str(node.id, node.data),
            },
        }
    elif inspect.isclass(node.data) and issubclass(node.data, BaseModel):
        json = (
            {
                "type": "schema",
                "data": node.data.schema(),
            }
            if with_schemas
            else {
                "type": "schema",
                "data": node_data_str(node.id, node.data),
            }
        )
    else:
        json = {
            "type": "unknown",
            "data": node_data_str(node.id, node.data),
        }
    if node.metadata is not None:
        json["metadata"] = node.metadata
    return json


@dataclass
class Graph:
    """Graph of nodes and edges."""

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def to_json(self, *, with_schemas: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Convert the graph to a JSON-serializable format."""
        stable_node_ids = {
            node.id: i if is_uuid(node.id) else node.id
            for i, node in enumerate(self.nodes.values())
        }
        edges: List[Dict[str, Any]] = []
        for edge in self.edges:
            edge_dict = {
                "source": stable_node_ids[edge.source],
                "target": stable_node_ids[edge.target],
            }
            if edge.data is not None:
                edge_dict["data"] = edge.data
            if edge.conditional:
                edge_dict["conditional"] = True
            edges.append(edge_dict)

        return {
            "nodes": [
                {
                    "id": stable_node_ids[node.id],
                    **node_data_json(node, with_schemas=with_schemas),
                }
                for node in self.nodes.values()
            ],
            "edges": edges,
        }

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def next_id(self) -> str:
        return uuid4().hex

    def add_node(
        self,
        data: Union[Type[BaseModel], RunnableType],
        id: Optional[str] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Node:
        """Add a node to the graph and return it."""
        if id is not None and id in self.nodes:
            raise ValueError(f"Node with id {id} already exists")
        id = id or self.next_id()
        node = Node(id=id, data=data, metadata=metadata, name=node_data_str(id, data))
        self.nodes[node.id] = node
        return node

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graph and all edges connected to it."""
        self.nodes.pop(node.id)
        self.edges = [
            edge
            for edge in self.edges
            if edge.source != node.id and edge.target != node.id
        ]

    def add_edge(
        self,
        source: Node,
        target: Node,
        data: Optional[Stringifiable] = None,
        conditional: bool = False,
    ) -> Edge:
        """Add an edge to the graph and return it."""
        if source.id not in self.nodes:
            raise ValueError(f"Source node {source.id} not in graph")
        if target.id not in self.nodes:
            raise ValueError(f"Target node {target.id} not in graph")
        edge = Edge(
            source=source.id, target=target.id, data=data, conditional=conditional
        )
        self.edges.append(edge)
        return edge

    def extend(
        self, graph: Graph, *, prefix: str = ""
    ) -> Tuple[Optional[Node], Optional[Node]]:
        """Add all nodes and edges from another graph.
        Note this doesn't check for duplicates, nor does it connect the graphs."""
        if all(is_uuid(node.id) for node in graph.nodes.values()):
            prefix = ""

        def prefixed(id: str) -> str:
            return f"{prefix}:{id}" if prefix else id

        # prefix each node
        self.nodes.update(
            {prefixed(k): v.copy(id=prefixed(k)) for k, v in graph.nodes.items()}
        )
        # prefix each edge's source and target
        self.edges.extend(
            [
                edge.copy(source=prefixed(edge.source), target=prefixed(edge.target))
                for edge in graph.edges
            ]
        )
        # return (prefixed) first and last nodes of the subgraph
        first, last = graph.first_node(), graph.last_node()
        return (
            first.copy(id=prefixed(first.id)) if first else None,
            last.copy(id=prefixed(last.id)) if last else None,
        )

    def reid(self) -> Graph:
        """Return a new graph with all nodes re-identified,
        using their unique, readable names where possible."""
        node_labels = {node.id: node.name for node in self.nodes.values()}
        node_label_counts = Counter(node_labels.values())

        def _get_node_id(node_id: str) -> str:
            label = node_labels[node_id]
            if is_uuid(node_id) and node_label_counts[label] == 1:
                return label
            else:
                return node_id

        return Graph(
            nodes={
                _get_node_id(id): node.copy(id=_get_node_id(id))
                for id, node in self.nodes.items()
            },
            edges=[
                edge.copy(
                    source=_get_node_id(edge.source),
                    target=_get_node_id(edge.target),
                )
                for edge in self.edges
            ],
        )

    def first_node(self) -> Optional[Node]:
        """Find the single node that is not a target of any edge.
        If there is no such node, or there are multiple, return None.
        When drawing the graph this node would be the origin."""
        targets = {edge.target for edge in self.edges}
        found: List[Node] = []
        for node in self.nodes.values():
            if node.id not in targets:
                found.append(node)
        return found[0] if len(found) == 1 else None

    def last_node(self) -> Optional[Node]:
        """Find the single node that is not a source of any edge.
        If there is no such node, or there are multiple, return None.
        When drawing the graph this node would be the destination.
        """
        sources = {edge.source for edge in self.edges}
        found: List[Node] = []
        for node in self.nodes.values():
            if node.id not in sources:
                found.append(node)
        return found[0] if len(found) == 1 else None

    def trim_first_node(self) -> None:
        """Remove the first node if it exists and has a single outgoing edge,
        ie. if removing it would not leave the graph without a "first" node."""
        first_node = self.first_node()
        if first_node:
            if (
                len(self.nodes) == 1
                or len([edge for edge in self.edges if edge.source == first_node.id])
                == 1
            ):
                self.remove_node(first_node)

    def trim_last_node(self) -> None:
        """Remove the last node if it exists and has a single incoming edge,
        ie. if removing it would not leave the graph without a "last" node."""
        last_node = self.last_node()
        if last_node:
            if (
                len(self.nodes) == 1
                or len([edge for edge in self.edges if edge.target == last_node.id])
                == 1
            ):
                self.remove_node(last_node)

    def draw_ascii(self) -> str:
        from langchain_core.runnables.graph_ascii import draw_ascii

        return draw_ascii(
            {node.id: node.name for node in self.nodes.values()},
            self.edges,
        )

    def print_ascii(self) -> None:
        print(self.draw_ascii())  # noqa: T201

    @overload
    def draw_png(
        self,
        output_file_path: str,
        fontname: Optional[str] = None,
        labels: Optional[LabelsDict] = None,
    ) -> None: ...

    @overload
    def draw_png(
        self,
        output_file_path: None,
        fontname: Optional[str] = None,
        labels: Optional[LabelsDict] = None,
    ) -> bytes: ...

    def draw_png(
        self,
        output_file_path: Optional[str] = None,
        fontname: Optional[str] = None,
        labels: Optional[LabelsDict] = None,
    ) -> Union[bytes, None]:
        from langchain_core.runnables.graph_png import PngDrawer

        default_node_labels = {node.id: node.name for node in self.nodes.values()}

        return PngDrawer(
            fontname,
            LabelsDict(
                nodes={
                    **default_node_labels,
                    **(labels["nodes"] if labels is not None else {}),
                },
                edges=labels["edges"] if labels is not None else {},
            ),
        ).draw(self, output_file_path)

    def draw_mermaid(
        self,
        *,
        with_styles: bool = True,
        curve_style: CurveStyle = CurveStyle.LINEAR,
        node_colors: NodeStyles = NodeStyles(),
        wrap_label_n_words: int = 9,
    ) -> str:
        from langchain_core.runnables.graph_mermaid import draw_mermaid

        graph = self.reid()
        first_node = graph.first_node()
        last_node = graph.last_node()

        return draw_mermaid(
            nodes=graph.nodes,
            edges=graph.edges,
            first_node=first_node.id if first_node else None,
            last_node=last_node.id if last_node else None,
            with_styles=with_styles,
            curve_style=curve_style,
            node_styles=node_colors,
            wrap_label_n_words=wrap_label_n_words,
        )

    def draw_mermaid_png(
        self,
        *,
        curve_style: CurveStyle = CurveStyle.LINEAR,
        node_colors: NodeStyles = NodeStyles(),
        wrap_label_n_words: int = 9,
        output_file_path: Optional[str] = None,
        draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
        background_color: str = "white",
        padding: int = 10,
    ) -> bytes:
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png

        mermaid_syntax = self.draw_mermaid(
            curve_style=curve_style,
            node_colors=node_colors,
            wrap_label_n_words=wrap_label_n_words,
        )
        return draw_mermaid_png(
            mermaid_syntax=mermaid_syntax,
            output_file_path=output_file_path,
            draw_method=draw_method,
            background_color=background_color,
            padding=padding,
        )
