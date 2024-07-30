from typing import TYPE_CHECKING, Dict, Iterable, Optional

from langchain_core.documents import Document
from langchain_core.graph_vectorstores.links import get_links

if TYPE_CHECKING:
    import graphviz


def _escape_id(id: str) -> str:
    return id.replace(":", "_")


_EDGE_DIRECTION = {
    "in": "back",
    "out": "forward",
    "bidir": "both",
}


def render_graphviz(
    documents: Iterable[Document],
    node_color: str = "white",
    node_colors: Optional[Dict[str, str]] = {},
) -> "graphviz.Digraph":
    """Render a collection of GraphVectorStore documents to GraphViz format.

    Args:
        documents: The documents to render.
        node_color: General node color. Defaults to `white`.
        node_colors: Dictionary specifying colors of specific nodes. Useful for
            emphasizing nodes that were selected by MMR, or differ from other
            results.

    Returns:
        The "graphviz.Digraph" representing the nodes. May be printed to source,
        or rendered using `dot`.

    Note:
        To render the generated DOT source code, you also need to install Graphviz_
        (`download page <https://www.graphviz.org/download/>`_,
        `archived versions <https://www2.graphviz.org/Archive/stable/>`_,
        `installation procedure for Windows <https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224>`_).
    """
    if node_colors is None:
        node_colors = {}
    try:
        import graphviz
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Could not import graphviz python package. "
            "Please install it with `pip install graphviz`."
        )

    try:
        graphviz.version()
    except graphviz.ExecutableNotFound:
        raise ImportError(
            "Could not execute `dot`. "
            "Make sure graphviz executable is installed (see https://www.graphviz.org/download/)."
        )

    tags = set()

    graph = graphviz.Digraph()
    graph.attr("node", style="filled")
    for document in documents:
        id = document.id
        if id is None:
            raise ValueError(f"Illegal graph document without ID: {document}")
        escaped_id = _escape_id(id)
        color = node_colors.get(id) or node_color
        graph.node(
            escaped_id,
            label=graphviz.escape(id),
            shape="note",
            fillcolor=color,
            tooltip=graphviz.escape(document.page_content),
        )

        for link in get_links(document):
            tag = f"{link.kind}_{link.tag}"
            if tag not in tags:
                graph.node(tag, label=graphviz.escape(f"{link.kind}:{link.tag}"))
                tags.add(tag)

            graph.edge(escaped_id, tag, dir=_EDGE_DIRECTION[link.direction])
    return graph
