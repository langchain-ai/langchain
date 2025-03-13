from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypedDict,
    Union,
    overload,
)
from uuid import UUID, uuid4

from langchain_core.utils.pydantic import _IgnoreUnserializable, is_basemodel_subclass

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

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
    except ValueError:
        return False
    return True


class Edge(NamedTuple):
    """Edge in a graph.

    Parameters:
        source: The source node id.
        target: The target node id.
        data: Optional data associated with the edge. Defaults to None.
        conditional: Whether the edge is conditional. Defaults to False.
    """

    source: str
    target: str
    data: Optional[Stringifiable] = None
    conditional: bool = False

    def copy(
        self, *, source: Optional[str] = None, target: Optional[str] = None
    ) -> Edge:
        """Return a copy of the edge with optional new source and target nodes.

        Args:
            source: The new source node id. Defaults to None.
            target: The new target node id. Defaults to None.

        Returns:
            A copy of the edge with the new source and target nodes.
        """
        return Edge(
            source=source or self.source,
            target=target or self.target,
            data=self.data,
            conditional=self.conditional,
        )


class Node(NamedTuple):
    """Node in a graph.

    Parameters:
        id: The unique identifier of the node.
        name: The name of the node.
        data: The data of the node.
        metadata: Optional metadata for the node. Defaults to None.
    """

    id: str
    name: str
    data: Union[type[BaseModel], RunnableType]
    metadata: Optional[dict[str, Any]]

    def copy(self, *, id: Optional[str] = None, name: Optional[str] = None) -> Node:
        """Return a copy of the node with optional new id and name.

        Args:
            id: The new node id. Defaults to None.
            name: The new node name. Defaults to None.

        Returns:
            A copy of the node with the new id and name.
        """
        return Node(
            id=id or self.id,
            name=name or self.name,
            data=self.data,
            metadata=self.metadata,
        )


class Branch(NamedTuple):
    """Branch in a graph.

    Parameters:
        condition: A callable that returns a string representation of the condition.
        ends: Optional dictionary of end node ids for the branches. Defaults
            to None.
    """

    condition: Callable[..., str]
    ends: Optional[dict[str, str]]


class CurveStyle(Enum):
    """Enum for different curve styles supported by Mermaid."""

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
    """Schema for Hexadecimal color codes for different node types.

    Parameters:
        default: The default color code. Defaults to "fill:#f2f0ff,line-height:1.2".
        first: The color code for the first node. Defaults to "fill-opacity:0".
        last: The color code for the last node. Defaults to "fill:#bfb6fc".
    """

    default: str = "fill:#f2f0ff,line-height:1.2"
    first: str = "fill-opacity:0"
    last: str = "fill:#bfb6fc"


class MermaidDrawMethod(Enum):
    """Enum for different draw methods supported by Mermaid."""

    PYPPETEER = "pyppeteer"  # Uses Pyppeteer to render the graph
    API = "api"  # Uses Mermaid.INK API to render the graph


def node_data_str(id: str, data: Union[type[BaseModel], RunnableType]) -> str:
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
) -> dict[str, Union[str, dict[str, Any]]]:
    """Convert the data of a node to a JSON-serializable format.

    Args:
        node: The node to convert.
        with_schemas: Whether to include the schema of the data if
            it is a Pydantic model. Defaults to False.

    Returns:
        A dictionary with the type of the data and the data itself.
    """
    from langchain_core.load.serializable import to_json_not_implemented
    from langchain_core.runnables.base import Runnable, RunnableSerializable

    if isinstance(node.data, RunnableSerializable):
        json: dict[str, Any] = {
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
    elif inspect.isclass(node.data) and is_basemodel_subclass(node.data):
        json = (
            {
                "type": "schema",
                "data": node.data.model_json_schema(
                    schema_generator=_IgnoreUnserializable
                ),
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
    """Graph of nodes and edges.

    Parameters:
        nodes: Dictionary of nodes in the graph. Defaults to an empty dictionary.
        edges: List of edges in the graph. Defaults to an empty list.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def to_json(self, *, with_schemas: bool = False) -> dict[str, list[dict[str, Any]]]:
        """Convert the graph to a JSON-serializable format.

        Args:
            with_schemas: Whether to include the schemas of the nodes if they are
                Pydantic models. Defaults to False.

        Returns:
            A dictionary with the nodes and edges of the graph.
        """
        stable_node_ids = {
            node.id: i if is_uuid(node.id) else node.id
            for i, node in enumerate(self.nodes.values())
        }
        edges: list[dict[str, Any]] = []
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
        """Return a new unique node
        identifier that can be used to add a node to the graph.
        """
        return uuid4().hex

    def add_node(
        self,
        data: Union[type[BaseModel], RunnableType],
        id: Optional[str] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Node:
        """Add a node to the graph and return it.

        Args:
            data: The data of the node.
            id: The id of the node. Defaults to None.
            metadata: Optional metadata for the node. Defaults to None.

        Returns:
            The node that was added to the graph.

        Raises:
            ValueError: If a node with the same id already exists.
        """
        if id is not None and id in self.nodes:
            msg = f"Node with id {id} already exists"
            raise ValueError(msg)
        id = id or self.next_id()
        node = Node(id=id, data=data, metadata=metadata, name=node_data_str(id, data))
        self.nodes[node.id] = node
        return node

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graph and all edges connected to it.

        Args:
            node: The node to remove.
        """
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
        """Add an edge to the graph and return it.

        Args:
            source: The source node of the edge.
            target: The target node of the edge.
            data: Optional data associated with the edge. Defaults to None.
            conditional: Whether the edge is conditional. Defaults to False.

        Returns:
            The edge that was added to the graph.

        Raises:
            ValueError: If the source or target node is not in the graph.
        """
        if source.id not in self.nodes:
            msg = f"Source node {source.id} not in graph"
            raise ValueError(msg)
        if target.id not in self.nodes:
            msg = f"Target node {target.id} not in graph"
            raise ValueError(msg)
        edge = Edge(
            source=source.id, target=target.id, data=data, conditional=conditional
        )
        self.edges.append(edge)
        return edge

    def extend(
        self, graph: Graph, *, prefix: str = ""
    ) -> tuple[Optional[Node], Optional[Node]]:
        """Add all nodes and edges from another graph.
        Note this doesn't check for duplicates, nor does it connect the graphs.

        Args:
            graph: The graph to add.
            prefix: The prefix to add to the node ids. Defaults to "".

        Returns:
            A tuple of the first and last nodes of the subgraph.
        """
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
        using their unique, readable names where possible.
        """
        node_name_to_ids = defaultdict(list)
        for node in self.nodes.values():
            node_name_to_ids[node.name].append(node.id)

        unique_labels = {
            node_id: node_name if len(node_ids) == 1 else f"{node_name}_{i + 1}"
            for node_name, node_ids in node_name_to_ids.items()
            for i, node_id in enumerate(node_ids)
        }

        def _get_node_id(node_id: str) -> str:
            label = unique_labels[node_id]
            if is_uuid(node_id):
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
        When drawing the graph, this node would be the origin.
        """
        return _first_node(self)

    def last_node(self) -> Optional[Node]:
        """Find the single node that is not a source of any edge.
        If there is no such node, or there are multiple, return None.
        When drawing the graph, this node would be the destination.
        """
        return _last_node(self)

    def trim_first_node(self) -> None:
        """Remove the first node if it exists and has a single outgoing edge,
        i.e., if removing it would not leave the graph without a "first" node.
        """
        first_node = self.first_node()
        if (
            first_node
            and _first_node(self, exclude=[first_node.id])
            and len({e for e in self.edges if e.source == first_node.id}) == 1
        ):
            self.remove_node(first_node)

    def trim_last_node(self) -> None:
        """Remove the last node if it exists and has a single incoming edge,
        i.e., if removing it would not leave the graph without a "last" node.
        """
        last_node = self.last_node()
        if (
            last_node
            and _last_node(self, exclude=[last_node.id])
            and len({e for e in self.edges if e.target == last_node.id}) == 1
        ):
            self.remove_node(last_node)

    def draw_ascii(self) -> str:
        """Draw the graph as an ASCII art string."""
        from langchain_core.runnables.graph_ascii import draw_ascii

        return draw_ascii(
            {node.id: node.name for node in self.nodes.values()},
            self.edges,
        )

    def print_ascii(self) -> None:
        """Print the graph as an ASCII art string."""
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
        """Draw the graph as a PNG image.

        Args:
            output_file_path: The path to save the image to. If None, the image
                is not saved. Defaults to None.
            fontname: The name of the font to use. Defaults to None.
            labels: Optional labels for nodes and edges in the graph. Defaults to None.

        Returns:
            The PNG image as bytes if output_file_path is None, None otherwise.
        """
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
        node_colors: Optional[NodeStyles] = None,
        wrap_label_n_words: int = 9,
    ) -> str:
        """Draw the graph as a Mermaid syntax string.

        Args:
            with_styles: Whether to include styles in the syntax. Defaults to True.
            curve_style: The style of the edges. Defaults to CurveStyle.LINEAR.
            node_colors: The colors of the nodes. Defaults to NodeStyles().
            wrap_label_n_words: The number of words to wrap the node labels at.
                Defaults to 9.

        Returns:
            The Mermaid syntax string.
        """
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
        node_colors: Optional[NodeStyles] = None,
        wrap_label_n_words: int = 9,
        output_file_path: Optional[str] = None,
        draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
        background_color: str = "white",
        padding: int = 10,
    ) -> bytes:
        """Draw the graph as a PNG image using Mermaid.

        Args:
            curve_style: The style of the edges. Defaults to CurveStyle.LINEAR.
            node_colors: The colors of the nodes. Defaults to NodeStyles().
            wrap_label_n_words: The number of words to wrap the node labels at.
                Defaults to 9.
            output_file_path: The path to save the image to. If None, the image
                is not saved. Defaults to None.
            draw_method: The method to use to draw the graph.
                Defaults to MermaidDrawMethod.API.
            background_color: The color of the background. Defaults to "white".
            padding: The padding around the graph. Defaults to 10.

        Returns:
            The PNG image as bytes.
        """
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


def _first_node(graph: Graph, exclude: Sequence[str] = ()) -> Optional[Node]:
    """Find the single node that is not a target of any edge.
    Exclude nodes/sources with ids in the exclude list.
    If there is no such node, or there are multiple, return None.
    When drawing the graph, this node would be the origin.
    """
    targets = {edge.target for edge in graph.edges if edge.source not in exclude}
    found: list[Node] = []
    for node in graph.nodes.values():
        if node.id not in exclude and node.id not in targets:
            found.append(node)
    return found[0] if len(found) == 1 else None


def _last_node(graph: Graph, exclude: Sequence[str] = ()) -> Optional[Node]:
    """Find the single node that is not a source of any edge.
    Exclude nodes/targets with ids in the exclude list.
    If there is no such node, or there are multiple, return None.
    When drawing the graph, this node would be the destination.
    """
    sources = {edge.source for edge in graph.edges if edge.target not in exclude}
    found: list[Node] = []
    for node in graph.nodes.values():
        if node.id not in exclude and node.id not in sources:
            found.append(node)
    return found[0] if len(found) == 1 else None
