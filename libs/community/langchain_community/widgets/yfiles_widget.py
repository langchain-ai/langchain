from typing import Any, Dict, Optional

from langchain_core.runnables.base import Runnable


class YfilesJupyterWidget:
    def show_jupyter_widget(
        self,
        runnable: Runnable,
        directed: bool = True,
        start: str = "#FFDFBA",
        end: str = "#BAFFC9",
        node_color=None,
    ):
        """
        NOTE: This function  requires yfiles_jupyter_graphs to be installed.
        NOTE: This method exclusively works in jupyter environments.
        """
        try:
            from yfiles_jupyter_graphs import GraphWidget
        except ImportError:
            try:
                from yfiles_jupyter_graphs import GraphWidget
            except ImportError:
                raise ImportError(
                    "Please install yfiles_jupyter_graphs to visualize the graph: `pip install yfiles_jupyter_graphs`"
                )

        w = GraphWidget()
        data = runnable.get_graph().to_json()
        nodes = data["nodes"]
        edges = data["edges"]
        for node in nodes:
            label = node["data"]["name"] if "name" in node["data"] else node["data"]
            w.nodes.append(
                {
                    "id": node["id"],
                    "properties": {
                        "type": node["type"],
                        "data": node["data"],
                        "label": label,
                    },
                }
            )
        for edge in edges:
            conditional = edge["conditional"] if "conditional" in edge else False
            w.edges.append(
                {
                    "start": edge["source"],
                    "end": edge["target"],
                    "properties": {"conditional": conditional},
                }
            )
        w.sidebar = False

        def color_mapping(node: Dict):
            if node["id"] == "__start__":
                return start
            elif node["id"] == "__end__":
                return end
            else:
                if node_color is not None:
                    return node_color

        w.set_node_color_mapping(color_mapping)
        w.directed = directed
        display(w)
