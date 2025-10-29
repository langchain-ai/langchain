"""Helper class to draw a state graph into a PNG file."""

from typing import Any

from langchain_core.runnables.graph import Graph, LabelsDict

try:
    import pygraphviz as pgv  # type: ignore[import-not-found]

    _HAS_PYGRAPHVIZ = True
except ImportError:
    _HAS_PYGRAPHVIZ = False


class PngDrawer:
    """Helper class to draw a state graph into a PNG file.

    It requires `graphviz` and `pygraphviz` to be installed.

    Example:
        ```python
        drawer = PngDrawer()
        drawer.draw(state_graph, "graph.png")
        ```
    """

    def __init__(
        self, fontname: str | None = None, labels: LabelsDict | None = None
    ) -> None:
        """Initializes the PNG drawer.

        Args:
            fontname: The font to use for the labels. Defaults to "arial".
            labels: A dictionary of label overrides. The dictionary
                should have the following format:
                {
                    "nodes": {
                        "node1": "CustomLabel1",
                        "node2": "CustomLabel2",
                        "__end__": "End Node"
                    },
                    "edges": {
                        "continue": "ContinueLabel",
                        "end": "EndLabel"
                    }
                }
                The keys are the original labels, and the values are the new labels.

        """
        self.fontname = fontname or "arial"
        self.labels = labels or LabelsDict(nodes={}, edges={})

    def get_node_label(self, label: str) -> str:
        """Returns the label to use for a node.

        Args:
            label: The original label.

        Returns:
            The new label.
        """
        label = self.labels.get("nodes", {}).get(label, label)
        return f"<<B>{label}</B>>"

    def get_edge_label(self, label: str) -> str:
        """Returns the label to use for an edge.

        Args:
            label: The original label.

        Returns:
            The new label.
        """
        label = self.labels.get("edges", {}).get(label, label)
        return f"<<U>{label}</U>>"

    def add_node(self, viz: Any, node: str) -> None:
        """Adds a node to the graph.

        Args:
            viz: The graphviz object.
            node: The node to add.
        """
        viz.add_node(
            node,
            label=self.get_node_label(node),
            style="filled",
            fillcolor="yellow",
            fontsize=15,
            fontname=self.fontname,
        )

    def add_edge(
        self,
        viz: Any,
        source: str,
        target: str,
        label: str | None = None,
        conditional: bool = False,  # noqa: FBT001,FBT002
    ) -> None:
        """Adds an edge to the graph.

        Args:
            viz: The graphviz object.
            source: The source node.
            target: The target node.
            label: The label for the edge.
            conditional: Whether the edge is conditional.
        """
        viz.add_edge(
            source,
            target,
            label=self.get_edge_label(label) if label else "",
            fontsize=12,
            fontname=self.fontname,
            style="dotted" if conditional else "solid",
        )

    def draw(self, graph: Graph, output_path: str | None = None) -> bytes | None:
        """Draw the given state graph into a PNG file.

        Requires `graphviz` and `pygraphviz` to be installed.

        Args:
            graph: The graph to draw
            output_path: The path to save the PNG. If `None`, PNG bytes are returned.

        Raises:
            ImportError: If `pygraphviz` is not installed.

        Returns:
            The PNG bytes if `output_path` is None, else None.
        """
        if not _HAS_PYGRAPHVIZ:
            msg = "Install pygraphviz to draw graphs: `pip install pygraphviz`."
            raise ImportError(msg)

        # Create a directed graph
        viz = pgv.AGraph(directed=True, nodesep=0.9, ranksep=1.0)

        # Add nodes, conditional edges, and edges to the graph
        self.add_nodes(viz, graph)
        self.add_edges(viz, graph)

        # Update entrypoint and END styles
        self.update_styles(viz, graph)

        # Save the graph as PNG
        try:
            return viz.draw(output_path, format="png", prog="dot")
        finally:
            viz.close()

    def add_nodes(self, viz: Any, graph: Graph) -> None:
        """Add nodes to the graph.

        Args:
            viz: The graphviz object.
            graph: The graph to draw.
        """
        for node in graph.nodes:
            self.add_node(viz, node)

    def add_edges(self, viz: Any, graph: Graph) -> None:
        """Add edges to the graph.

        Args:
            viz: The graphviz object.
            graph: The graph to draw.
        """
        for start, end, data, cond in graph.edges:
            self.add_edge(
                viz, start, end, str(data) if data is not None else None, cond
            )

    def update_styles(self, viz: Any, graph: Graph) -> None:
        """Update the styles of the entrypoint and END nodes.

        Args:
            viz: The graphviz object.
            graph: The graph to draw.
        """
        if first := graph.first_node():
            viz.get_node(first.id).attr.update(fillcolor="lightblue")
        if last := graph.last_node():
            viz.get_node(last.id).attr.update(fillcolor="orange")
