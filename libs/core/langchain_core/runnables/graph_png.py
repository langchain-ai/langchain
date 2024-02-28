from typing import Optional

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

    def __init__(self, fontname="arial", labels: Optional[LabelsDict] = None):
        self.fontname = fontname
        self.labels = labels or LabelsDict(nodes={}, edges={})

    def get_node_label(self, label: str) -> str:
        label = self.labels.get("nodes", {}).get(label, label)
        return f"<<B>{label}</B>>"

    def get_edge_label(self, label: str) -> str:
        label = self.labels.get("edges", {}).get(label, label)
        return f"<<U>{label}</U>>"

    def add_node(self, graphviz_graph, node: str, label: str = None):
        if not label:
            label = node

        graphviz_graph.add_node(
            node,
            label=self.get_node_label(label),
            style="filled",
            fillcolor="yellow",
            fontsize=15,
            fontname=self.fontname,
        )

    def add_edge(self, graphviz_graph, source: str, target: str, label: str = None):
        graphviz_graph.add_edge(
            source,
            target,
            label=self.get_edge_label(label) if label else "",
            fontsize=12,
            fontname=self.fontname,
        )

    def draw(self, state_graph: Graph, output_file_path="graph.png"):
        """
        Draws the given state graph into a PNG file.
        Requires graphviz and pygraphviz to be installed.
        :param state_graph: The state graph to draw
        :param output_file_path: The path to the output file
        """

        try:
            import pygraphviz as pgv
        except ImportError as exc:
            raise ImportError(
                "Install pygraphviz to draw graphs: `pip install pygraphviz`."
            ) from exc

        # Create a directed graph
        graphviz_graph = pgv.AGraph(directed=True, nodesep=0.9, ranksep=1.0)

        # Add nodes, conditional edges, and edges to the graph
        self.add_nodes(graphviz_graph, state_graph)
        self.add_edges(graphviz_graph, state_graph)

        # Update entrypoint and END styles
        self.update_styles(graphviz_graph, state_graph)

        # Save the graph as PNG
        graphviz_graph.draw(output_file_path, format="png", prog="dot")
        graphviz_graph.close()

    def add_nodes(self, viz, graph: Graph):
        for node in graph.nodes:
            self.add_node(viz, node, node)

    def add_edges(self, viz, graph: Graph):
        for start, end, label in graph.edges:
            self.add_edge(viz, start, end, label)

    def update_styles(self, viz, graph: Graph):
        if first := graph.first_node():
            viz.get_node(first.id).attr.update(fillcolor="lightblue")
        if last := graph.last_node():
            viz.get_node(last.id).attr.update(fillcolor="orange")
