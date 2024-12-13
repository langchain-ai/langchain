"""Utilities for using Graph Vector Stores with networkx."""

import typing

from langchain_core.documents import Document

from langchain_community.graph_vectorstores.links import get_links

if typing.TYPE_CHECKING:
    import networkx as nx


def documents_to_networkx(
    documents: typing.Iterable[Document],
    *,
    tag_nodes: bool = True,
) -> "nx.DiGraph":
    """Return the networkx directed graph corresponding to the documents.

    Args:
        documents: The documents to convenrt to networkx.
        tag_nodes: If `True`, each tag will be rendered as a node, with edges
            to/from the corresponding documents. If `False`, edges will be
            between documents, with a label corresponding to the tag(s)
            connecting them.
    """
    import networkx as nx

    graph = nx.DiGraph()

    tag_ids: typing.Dict[typing.Tuple[str, str], str] = {}
    tag_labels: typing.Dict[str, str] = {}
    documents_by_incoming: typing.Dict[str, typing.Set[str]] = {}

    # First pass:
    #   - Register tag IDs for each unique (kind, tag).
    #   - If rendering tag nodes, add them to the graph.
    #   - If not rendering tag nodes, create a dictionary of documents by incoming tags.
    for document in documents:
        if document.id is None:
            raise ValueError(f"Illegal graph document without ID: {document}")

        for link in get_links(document):
            tag_key = (link.kind, link.tag)
            tag_id = tag_ids.get(tag_key)
            if tag_id is None:
                tag_id = f"tag_{len(tag_ids)}"
                tag_ids[tag_key] = tag_id

                if tag_nodes:
                    graph.add_node(tag_id, label=f"{link.kind}:{link.tag}")

            if not tag_nodes and (link.direction == "in" or link.direction == "bidir"):
                tag_labels[tag_id] = f"{link.kind}:{link.tag}"
                documents_by_incoming.setdefault(tag_id, set()).add(document.id)

    # Second pass:
    #   - Render document nodes
    #   - If rendering tag nodes, render edges to/from documents and tag nodes.
    #   - If not rendering tag nodes, render edges to/from documents based on tags.
    for document in documents:
        graph.add_node(document.id, text=document.page_content)

        targets: typing.Dict[str, typing.List[str]] = {}
        for link in get_links(document):
            tag_id = tag_ids[(link.kind, link.tag)]
            if tag_nodes:
                if link.direction == "in" or link.direction == "bidir":
                    graph.add_edge(tag_id, document.id)
                if link.direction == "out" or link.direction == "bidir":
                    graph.add_edge(document.id, tag_id)
            else:
                if link.direction == "out" or link.direction == "bidir":
                    label = tag_labels[tag_id]
                    for target in documents_by_incoming[tag_id]:
                        if target != document.id:
                            targets.setdefault(target, []).append(label)

        # Avoid a multigraph by collecting the list of labels for each edge.
        if not tag_nodes:
            for target, labels in targets.items():
                graph.add_edge(document.id, target, label=str(labels))

    return graph
