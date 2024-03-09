from typing import Any, Optional

from langchain_core.runnables.graph import Graph, LabelsDict


class PngDrawer:
    """
    A helper class to draw a state graph into a PNG file.
    Requires graphviz and pygraphviz to be installed.
    :param fontname: The font to use for the labels
    :param labels: A dictionary of label overrides. The dictionary
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
    Usage:
        drawer = PngDrawer()
        drawer.draw(state_graph, 'graph.png')
    """

    def __init__(
        self, fontname: Optional[str] = None, labels: Optional[LabelsDict] = None
    ) -> None:
        self.fontname = fontname or "arial"
        self.labels = labels or LabelsDict(nodes={}, edges={})

    def get_node_label(self, label: str) -> str:
        label = self.labels.get("nodes", {}).get(label, label)
        return f"<<B>{label}</B>>"

    def get_edge_label(self, label: str) -> str:
        label = self.labels.get("edges", {}).get(label, label)
        return f"<<U>{label}</U>>"

    def add_node(self, viz: Any, node: str) -> None:
        viz.add_node(
            node,
            label=self.get_node_label(node),
            style="filled",
            fillcolor="yellow",
            fontsize=15,
            fontname=self.fontname,
        )

    def add_edge(
        self, viz: Any, source: str, target: str, label: Optional[str] = None
    ) -> None:
        viz.add_edge(
            source,
            target,
            label=self.get_edge_label(label) if label else "",
            fontsize=12,
            fontname=self.fontname,
        )

    def draw(self, graph: Graph, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Draws the given state graph into a PNG file.
        Requires graphviz and pygraphviz to be installed.
        :param graph: The graph to draw
        :param output_path: The path to save the PNG. If None, PNG bytes are returned.
        """

        try:
            import pygraphviz as pgv  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install pygraphviz to draw graphs: `pip install pygraphviz`."
            ) from exc

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
        for node in graph.nodes:
            self.add_node(viz, node)

    def add_edges(self, viz: Any, graph: Graph) -> None:
        for start, end, label in graph.edges:
            self.add_edge(viz, start, end, label)

    def update_styles(self, viz: Any, graph: Graph) -> None:
        if first := graph.first_node():
            viz.get_node(first.id).attr.update(fillcolor="lightblue")
        if last := graph.last_node():
            viz.get_node(last.id).attr.update(fillcolor="orange")
