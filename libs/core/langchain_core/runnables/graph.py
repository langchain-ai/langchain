"""Graph used in `Runnable` objects."""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypedDict,
    overload,
)
from uuid import UUID, uuid4

from langchain_core.load.serializable import to_json_not_implemented
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.utils.pydantic import _IgnoreUnserializable, is_basemodel_subclass

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic import BaseModel

    from langchain_core.runnables.base import Runnable as RunnableType


class Stringifiable(Protocol):
    """Protocol for objects that can be converted to a string."""

    def __str__(self) -> str:
        """Convert the object to a string."""


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
        `True` if the string is a valid UUID, `False` otherwise.
    """
    try:
        UUID(value)
    except ValueError:
        return False
    return True


class Edge(NamedTuple):
    """Edge in a graph."""

    source: str
    """The source node id."""
    target: str
    """The target node id."""
    data: Stringifiable | None = None
    """Optional data associated with the edge. """
    conditional: bool = False
    """Whether the edge is conditional."""

    def copy(self, *, source: str | None = None, target: str | None = None) -> Edge:
        """Return a copy of the edge with optional new source and target nodes.

        Args:
            source: The new source node id.
            target: The new target node id.

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
    """Node in a graph."""

    id: str
    """The unique identifier of the node."""
    name: str
    """The name of the node."""
    data: type[BaseModel] | RunnableType | None
    """The data of the node."""
    metadata: dict[str, Any] | None
    """Optional metadata for the node. """

    def copy(
        self,
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> Node:
        """Return a copy of the node with optional new id and name.

        Args:
            id: The new node id.
            name: The new node name.

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
    """Branch in a graph."""

    condition: Callable[..., str]
    """A callable that returns a string representation of the condition."""
    ends: dict[str, str] | None
    """Optional dictionary of end node IDs for the branches. """


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

    Args:
        default: The default color code.
        first: The color code for the first node.
        last: The color code for the last node.
    """

    default: str = "fill:#f2f0ff,line-height:1.2"
    first: str = "fill-opacity:0"
    last: str = "fill:#bfb6fc"


class MermaidDrawMethod(Enum):
    """Enum for different draw methods supported by Mermaid."""

    PYPPETEER = "pyppeteer"
    """Uses Pyppeteer to render the graph"""
    API = "api"
    """Uses Mermaid.INK API to render the graph"""


def node_data_str(
    id: str,
    data: type[BaseModel] | RunnableType | None,
) -> str:
    """Convert the data of a node to a string.

    Args:
        id: The node id.
        data: The node data.

    Returns:
        A string representation of the data.
    """
    if not is_uuid(id) or data is None:
        return id
    data_str = data.get_name() if isinstance(data, Runnable) else data.__name__
    return data_str if not data_str.startswith("Runnable") else data_str[8:]


def node_data_json(
    node: Node, *, with_schemas: bool = False
) -> dict[str, str | dict[str, Any]]:
    """Convert the data of a node to a JSON-serializable format.

    Args:
        node: The `Node` to convert.
        with_schemas: Whether to include the schema of the data if it is a Pydantic
            model.

    Returns:
        A dictionary with the type of the data and the data itself.
    """
    if node.data is None:
        json: dict[str, Any] = {}
    elif isinstance(node.data, RunnableSerializable):
        json = {
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

    Args:
        nodes: Dictionary of nodes in the graph. Defaults to an empty dictionary.
        edges: List of edges in the graph. Defaults to an empty list.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def to_json(self, *, with_schemas: bool = False) -> dict[str, list[dict[str, Any]]]:
        """Convert the graph to a JSON-serializable format.

        Args:
            with_schemas: Whether to include the schemas of the nodes if they are
                Pydantic models.

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
                edge_dict["data"] = edge.data  # type: ignore[assignment]
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
        """Return whether the graph has any nodes."""
        return bool(self.nodes)

    def next_id(self) -> str:
        """Return a new unique node identifier.

        It that can be used to add a node to the graph.
        """
        return uuid4().hex

    def add_node(
        self,
        data: type[BaseModel] | RunnableType | None,
        id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Node:
        """Add a node to the graph and return it.

        Args:
            data: The data of the node.
            id: The id of the node.
            metadata: Optional metadata for the node.

        Returns:
            The node that was added to the graph.

        Raises:
            ValueError: If a node with the same id already exists.
        """
        if id is not None and id in self.nodes:
            msg = f"Node with id {id} already exists"
            raise ValueError(msg)
        id_ = id or self.next_id()
        node = Node(id=id_, data=data, metadata=metadata, name=node_data_str(id_, data))
        self.nodes[node.id] = node
        return node

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graph and all edges connected to it.

        Args:
            node: The node to remove.
        """
        self.nodes.pop(node.id)
        self.edges = [
            edge for edge in self.edges if node.id not in {edge.source, edge.target}
        ]

    def add_edge(
        self,
        source: Node,
        target: Node,
        data: Stringifiable | None = None,
        conditional: bool = False,  # noqa: FBT001,FBT002
    ) -> Edge:
        """Add an edge to the graph and return it.

        Args:
            source: The source node of the edge.
            target: The target node of the edge.
            data: Optional data associated with the edge.
            conditional: Whether the edge is conditional.

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
    ) -> tuple[Node | None, Node | None]:
        """Add all nodes and edges from another graph.

        Note this doesn't check for duplicates, nor does it connect the graphs.

        Args:
            graph: The graph to add.
            prefix: The prefix to add to the node ids.

        Returns:
            A tuple of the first and last nodes of the subgraph.
        """
        if all(is_uuid(node.id) for node in graph.nodes.values()):
            prefix = ""

        def prefixed(id_: str) -> str:
            return f"{prefix}:{id_}" if prefix else id_

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
        """Return a new graph with all nodes re-identified.

        Uses their unique, readable names where possible.
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
            return node_id

        return Graph(
            nodes={
                _get_node_id(id_): node.copy(id=_get_node_id(id_))
                for id_, node in self.nodes.items()
            },
            edges=[
                edge.copy(
                    source=_get_node_id(edge.source),
                    target=_get_node_id(edge.target),
                )
                for edge in self.edges
            ],
        )

    def first_node(self) -> Node | None:
        """Find the single node that is not a target of any edge.

        If there is no such node, or there are multiple, return `None`.
        When drawing the graph, this node would be the origin.

        Returns:
            The first node, or None if there is no such node or multiple
            candidates.
        """
        return _first_node(self)

    def last_node(self) -> Node | None:
        """Find the single node that is not a source of any edge.

        If there is no such node, or there are multiple, return `None`.
        When drawing the graph, this node would be the destination.

        Returns:
            The last node, or None if there is no such node or multiple
            candidates.
        """
        return _last_node(self)

    def trim_first_node(self) -> None:
        """Remove the first node if it exists and has a single outgoing edge.

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
        """Remove the last node if it exists and has a single incoming edge.

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
        """Draw the graph as an ASCII art string.

        Returns:
            The ASCII art string.
        """
        # Import locally to prevent circular import
        from langchain_core.runnables.graph_ascii import draw_ascii  # noqa: PLC0415

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
        fontname: str | None = None,
        labels: LabelsDict | None = None,
    ) -> None: ...

    @overload
    def draw_png(
        self,
        output_file_path: None,
        fontname: str | None = None,
        labels: LabelsDict | None = None,
    ) -> bytes: ...

    def draw_png(
        self,
        output_file_path: str | None = None,
        fontname: str | None = None,
        labels: LabelsDict | None = None,
    ) -> bytes | None:
        """Draw the graph as a PNG image.

        Args:
            output_file_path: The path to save the image to. If `None`, the image
                is not saved.
            fontname: The name of the font to use.
            labels: Optional labels for nodes and edges in the graph. Defaults to
                `None`.

        Returns:
            The PNG image as bytes if output_file_path is None, None otherwise.
        """
        # Import locally to prevent circular import
        from langchain_core.runnables.graph_png import PngDrawer  # noqa: PLC0415

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
        node_colors: NodeStyles | None = None,
        wrap_label_n_words: int = 9,
        frontmatter_config: dict[str, Any] | None = None,
    ) -> str:
        """Draw the graph as a Mermaid syntax string.

        Args:
            with_styles: Whether to include styles in the syntax.
            curve_style: The style of the edges.
            node_colors: The colors of the nodes.
            wrap_label_n_words: The number of words to wrap the node labels at.
            frontmatter_config: Mermaid frontmatter config.
                Can be used to customize theme and styles. Will be converted to YAML and
                added to the beginning of the mermaid graph.

                See more here: https://mermaid.js.org/config/configuration.html.

                Example config:

                ```python
                {
                    "config": {
                        "theme": "neutral",
                        "look": "handDrawn",
                        "themeVariables": {"primaryColor": "#e2e2e2"},
                    }
                }
                ```
        Returns:
            The Mermaid syntax string.
        """
        # Import locally to prevent circular import
        from langchain_core.runnables.graph_mermaid import draw_mermaid  # noqa: PLC0415

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
            frontmatter_config=frontmatter_config,
        )

    def draw_mermaid_png(
        self,
        *,
        curve_style: CurveStyle = CurveStyle.LINEAR,
        node_colors: NodeStyles | None = None,
        wrap_label_n_words: int = 9,
        output_file_path: str | None = None,
        draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
        background_color: str = "white",
        padding: int = 10,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        frontmatter_config: dict[str, Any] | None = None,
        base_url: str | None = None,
        proxies: dict[str, str] | None = None,
    ) -> bytes:
        """Draw the graph as a PNG image using Mermaid.

        Args:
            curve_style: The style of the edges.
            node_colors: The colors of the nodes.
            wrap_label_n_words: The number of words to wrap the node labels at.
            output_file_path: The path to save the image to. If `None`, the image
                is not saved.
            draw_method: The method to use to draw the graph.
            background_color: The color of the background.
            padding: The padding around the graph.
            max_retries: The maximum number of retries (`MermaidDrawMethod.API`).
            retry_delay: The delay between retries (`MermaidDrawMethod.API`).
            frontmatter_config: Mermaid frontmatter config.
                Can be used to customize theme and styles. Will be converted to YAML and
                added to the beginning of the mermaid graph.

                See more here: https://mermaid.js.org/config/configuration.html.

                Example config:

                ```python
                {
                    "config": {
                        "theme": "neutral",
                        "look": "handDrawn",
                        "themeVariables": {"primaryColor": "#e2e2e2"},
                    }
                }
                ```
            base_url: The base URL of the Mermaid server for rendering via API.
            proxies: HTTP/HTTPS proxies for requests (e.g. `{"http": "http://127.0.0.1:7890"}`).

        Returns:
            The PNG image as bytes.
        """
        # Import locally to prevent circular import
        from langchain_core.runnables.graph_mermaid import (  # noqa: PLC0415
            draw_mermaid_png,
        )

        mermaid_syntax = self.draw_mermaid(
            curve_style=curve_style,
            node_colors=node_colors,
            wrap_label_n_words=wrap_label_n_words,
            frontmatter_config=frontmatter_config,
        )
        return draw_mermaid_png(
            mermaid_syntax=mermaid_syntax,
            output_file_path=output_file_path,
            draw_method=draw_method,
            background_color=background_color,
            padding=padding,
            max_retries=max_retries,
            retry_delay=retry_delay,
            proxies=proxies,
            base_url=base_url,
        )


def _first_node(graph: Graph, exclude: Sequence[str] = ()) -> Node | None:
    """Find the single node that is not a target of any edge.

    Exclude nodes/sources with IDs in the exclude list.

    If there is no such node, or there are multiple, return `None`.

    When drawing the graph, this node would be the origin.
    """
    targets = {edge.target for edge in graph.edges if edge.source not in exclude}
    found: list[Node] = [
        node
        for node in graph.nodes.values()
        if node.id not in exclude and node.id not in targets
    ]
    return found[0] if len(found) == 1 else None


def _last_node(graph: Graph, exclude: Sequence[str] = ()) -> Node | None:
    """Find the single node that is not a source of any edge.

    Exclude nodes/targets with IDs in the exclude list.

    If there is no such node, or there are multiple, return `None`.

    When drawing the graph, this node would be the destination.
    """
    sources = {edge.source for edge in graph.edges if edge.target not in exclude}
    found: list[Node] = [
        node
        for node in graph.nodes.values()
        if node.id not in exclude and node.id not in sources
    ]
    return found[0] if len(found) == 1 else None
