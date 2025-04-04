import pytest
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.links import METADATA_LINKS_KEY, Link
from langchain_community.graph_vectorstores.visualize import render_graphviz


@pytest.mark.requires("graphviz")
def test_visualize_simple_graph() -> None:
    doc1 = Document(
        id="a",
        page_content="some content",
        metadata={
            METADATA_LINKS_KEY: [
                Link.incoming("href", "a"),
                Link.bidir("kw", "foo"),
            ]
        },
    )
    doc2 = Document(
        id="b",
        page_content="<some\n more content>",
        metadata={
            METADATA_LINKS_KEY: [
                Link.incoming("href", "b"),
                Link.outgoing("href", "a"),
                Link.bidir("kw", "foo"),
                Link.bidir("kw", "bar"),
            ]
        },
    )

    assert render_graphviz([doc1, doc2]).source == (
        "digraph {\n"
        "\trankdir=LR\n"
        "\tnode [style=filled]\n"
        '\ta [label="a\nsome content" shape=note tooltip="some content"]\n'
        '\ttag_0 [label="href:a"]\n'
        "\ta -> tag_0 [dir=back]\n"
        '\ttag_1 [label="kw:foo"]\n'
        "\ta -> tag_1 [dir=both]\n"
        '\tb [label="b\n<some\n more content>" '
        'shape=note tooltip="<some\n more content>"]\n'
        '\ttag_2 [label="href:b"]\n'
        "\tb -> tag_2 [dir=back]\n"
        "\tb -> tag_0 [dir=forward]\n"
        "\tb -> tag_1 [dir=both]\n"
        '\ttag_3 [label="kw:bar"]\n'
        "\tb -> tag_3 [dir=both]\n"
        "}\n"
    )

    assert render_graphviz([doc1, doc2], engine="fdp").engine == "fdp"

    assert render_graphviz([doc1, doc2], node_colors={"a": "gold"}).source == (
        "digraph {\n"
        "\trankdir=LR\n"
        "\tnode [style=filled]\n"
        '\ta [label="a\nsome content" fillcolor=gold '
        'shape=note tooltip="some content"]\n'
        '\ttag_0 [label="href:a"]\n'
        "\ta -> tag_0 [dir=back]\n"
        '\ttag_1 [label="kw:foo"]\n'
        "\ta -> tag_1 [dir=both]\n"
        '\tb [label="b\n<some\n more content>" '
        'shape=note tooltip="<some\n more content>"]\n'
        '\ttag_2 [label="href:b"]\n'
        "\tb -> tag_2 [dir=back]\n"
        "\tb -> tag_0 [dir=forward]\n"
        "\tb -> tag_1 [dir=both]\n"
        '\ttag_3 [label="kw:bar"]\n'
        "\tb -> tag_3 [dir=both]\n"
        "}\n"
    )

    assert render_graphviz(
        [doc1, doc2], node_color="gold", node_colors={"a": None}
    ).source == (
        "digraph {\n"
        "\trankdir=LR\n"
        "\tnode [style=filled]\n"
        '\ta [label="a\nsome content" shape=note tooltip="some content"]\n'
        '\ttag_0 [label="href:a"]\n'
        "\ta -> tag_0 [dir=back]\n"
        '\ttag_1 [label="kw:foo"]\n'
        "\ta -> tag_1 [dir=both]\n"
        '\tb [label="b\n<some\n more content>" fillcolor=gold '
        'shape=note tooltip="<some\n more content>"]\n'
        '\ttag_2 [label="href:b"]\n'
        "\tb -> tag_2 [dir=back]\n"
        "\tb -> tag_0 [dir=forward]\n"
        "\tb -> tag_1 [dir=both]\n"
        '\ttag_3 [label="kw:bar"]\n'
        "\tb -> tag_3 [dir=both]\n"
        "}\n"
    )

    assert render_graphviz([doc1, doc2], skip_tags=[("kw", "foo")]).source == (
        "digraph {\n"
        "\trankdir=LR\n"
        "\tnode [style=filled]\n"
        '\ta [label="a\nsome content" shape=note tooltip="some content"]\n'
        '\ttag_0 [label="href:a"]\n'
        "\ta -> tag_0 [dir=back]\n"
        '\tb [label="b\n<some\n more content>" '
        'shape=note tooltip="<some\n more content>"]\n'
        '\ttag_1 [label="href:b"]\n'
        "\tb -> tag_1 [dir=back]\n"
        "\tb -> tag_0 [dir=forward]\n"
        '\ttag_2 [label="kw:bar"]\n'
        "\tb -> tag_2 [dir=both]\n"
        "}\n"
    )
