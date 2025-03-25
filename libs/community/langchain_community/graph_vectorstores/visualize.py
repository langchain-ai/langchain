import re
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

from langchain_core._api import beta
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.links import get_links

if TYPE_CHECKING:
    import graphviz


def _escape_id(id: str) -> str:
    return id.replace(":", "_")


_EDGE_DIRECTION = {
    "in": "back",
    "out": "forward",
    "bidir": "both",
}

_WORD_RE = re.compile(r"\s*\S+")


def _split_prefix(s: str, max_chars: int = 50) -> str:
    words = _WORD_RE.finditer(s)

    split = min(len(s), max_chars)
    for word in words:
        if word.end(0) > max_chars:
            break
        split = word.end(0)

    if split == len(s):
        return s
    else:
        return f"{s[0:split]}..."


@beta()
def render_graphviz(
    documents: Iterable[Document],
    engine: Optional[str] = None,
    node_color: Optional[str] = None,
    node_colors: Optional[Dict[str, Optional[str]]] = None,
    skip_tags: Iterable[Tuple[str, str]] = (),
) -> "graphviz.Digraph":
    """Render a collection of GraphVectorStore documents to GraphViz format.

    Args:
        documents: The documents to render.
        engine: GraphViz layout engine to use. `None` uses the default.
        node_color: Default node color.
        node_colors: Dictionary specifying colors of specific nodes. Useful for
            emphasizing nodes that were selected by MMR, or differ from other
            results.
        skip_tags: Set of tags to skip when rendering the graph. Specified as
            tuples containing the kind and tag.

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

    graph = graphviz.Digraph(engine=engine)
    graph.attr(rankdir="LR")
    graph.attr("node", style="filled")

    skip_tags = set(skip_tags)
    tags: dict[Tuple[str, str], str] = {}

    for document in documents:
        id = document.id
        if id is None:
            raise ValueError(f"Illegal graph document without ID: {document}")
        escaped_id = _escape_id(id)
        color = node_colors[id] if id in node_colors else node_color

        node_label = "\n".join(
            [
                graphviz.escape(id),
                graphviz.escape(_split_prefix(document.page_content)),
            ]
        )
        graph.node(
            escaped_id,
            label=node_label,
            shape="note",
            fillcolor=color,
            tooltip=graphviz.escape(document.page_content),
        )

        for link in get_links(document):
            tag_key = (link.kind, link.tag)
            if tag_key in skip_tags:
                continue

            tag_id = tags.get(tag_key)
            if tag_id is None:
                tag_id = f"tag_{len(tags)}"
                tags[tag_key] = tag_id
                graph.node(tag_id, label=graphviz.escape(f"{link.kind}:{link.tag}"))

            graph.edge(escaped_id, tag_id, dir=_EDGE_DIRECTION[link.direction])
    return graph
