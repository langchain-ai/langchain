from __future__ import annotations

import inspect
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
    data: Optional[str] = None
    conditional: bool = False


class Node(NamedTuple):
    """Node in a graph."""

    id: str
    data: Union[Type[BaseModel], RunnableType]


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
class NodeColors:
    """Schema for Hexadecimal color codes for different node types"""

    start: str = "#ffdfba"
    end: str = "#baffc9"
    other: str = "#fad7de"


class MermaidDrawMethod(Enum):
    """Enum for different draw methods supported by Mermaid"""

    PYPPETEER = "pyppeteer"  # Uses Pyppeteer to render the graph
    API = "api"  # Uses Mermaid.INK API to render the graph


def node_data_str(node: Node) -> str:
    """Convert the data of a node to a string.

    Args:
        node: The node to convert.

    Returns:
        A string representation of the data.
    """
    from langchain_core.runnables.base import Runnable

    if not is_uuid(node.id):
        return node.id
    elif isinstance(node.data, Runnable):
        try:
            data = str(node.data)
            if (
                data.startswith("<")
                or data[0] != data[0].upper()
                or len(data.splitlines()) > 1
            ):
                data = node.data.__class__.__name__
            elif len(data) > 42:
                data = data[:42] + "..."
        except Exception:
            data = node.data.__class__.__name__
    else:
        data = node.data.__name__
    return data if not data.startswith("Runnable") else data[8:]


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
        return {
            "type": "runnable",
            "data": {
                "id": node.data.lc_id(),
                "name": node.data.get_name(),
            },
        }
    elif isinstance(node.data, Runnable):
        return {
            "type": "runnable",
            "data": {
                "id": to_json_not_implemented(node.data)["id"],
                "name": node.data.get_name(),
            },
        }
    elif inspect.isclass(node.data) and issubclass(node.data, BaseModel):
        return (
            {
                "type": "schema",
                "data": node.data.schema(),
            }
            if with_schemas
            else {
                "type": "schema",
                "data": node_data_str(node),
            }
        )
    else:
        return {
            "type": "unknown",
            "data": node_data_str(node),
        }


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
        self, data: Union[Type[BaseModel], RunnableType], id: Optional[str] = None
    ) -> Node:
        """Add a node to the graph and return it."""
        if id is not None and id in self.nodes:
            raise ValueError(f"Node with id {id} already exists")
        node = Node(id=id or self.next_id(), data=data)
        self.nodes[node.id] = node
        return node

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graphm and all edges connected to it."""
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
        data: Optional[str] = None,
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
            {prefixed(k): Node(prefixed(k), v.data) for k, v in graph.nodes.items()}
        )
        # prefix each edge's source and target
        self.edges.extend(
            [
                Edge(
                    prefixed(edge.source),
                    prefixed(edge.target),
                    edge.data,
                    edge.conditional,
                )
                for edge in graph.edges
            ]
        )
        # return (prefixed) first and last nodes of the subgraph
        first, last = graph.first_node(), graph.last_node()
        return (
            Node(prefixed(first.id), first.data) if first else None,
            Node(prefixed(last.id), last.data) if last else None,
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
            {node.id: node_data_str(node) for node in self.nodes.values()},
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
    ) -> None:
        ...

    @overload
    def draw_png(
        self,
        output_file_path: None,
        fontname: Optional[str] = None,
        labels: Optional[LabelsDict] = None,
    ) -> bytes:
        ...

    def draw_png(
        self,
        output_file_path: Optional[str] = None,
        fontname: Optional[str] = None,
        labels: Optional[LabelsDict] = None,
    ) -> Union[bytes, None]:
        from langchain_core.runnables.graph_png import PngDrawer

        default_node_labels = {
            node.id: node_data_str(node) for node in self.nodes.values()
        }

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
        node_colors: NodeColors = NodeColors(
            start="#ffdfba", end="#baffc9", other="#fad7de"
        ),
        wrap_label_n_words: int = 9,
    ) -> str:
        from langchain_core.runnables.graph_mermaid import draw_mermaid

        nodes = {node.id: node_data_str(node) for node in self.nodes.values()}

        first_node = self.first_node()
        first_label = node_data_str(first_node) if first_node is not None else None

        last_node = self.last_node()
        last_label = node_data_str(last_node) if last_node is not None else None

        return draw_mermaid(
            nodes=nodes,
            edges=self.edges,
            first_node_label=first_label,
            last_node_label=last_label,
            with_styles=with_styles,
            curve_style=curve_style,
            node_colors=node_colors,
            wrap_label_n_words=wrap_label_n_words,
        )

    def draw_mermaid_png(
        self,
        *,
        curve_style: CurveStyle = CurveStyle.LINEAR,
        node_colors: NodeColors = NodeColors(
            start="#ffdfba", end="#baffc9", other="#fad7de"
        ),
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
