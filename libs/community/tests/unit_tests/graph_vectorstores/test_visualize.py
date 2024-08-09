import pytest
from langchain_core.documents import Document
from langchain_core.graph_vectorstores.links import METADATA_LINKS_KEY, Link

from langchain_community.graph_vectorstores.visualize import render_graphviz


@pytest.mark.requires("graphviz")
def test_visualize_simple_graph():
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
        '\thref_a [label="href:a"]\n'
        "\ta -> href_a [dir=back]\n"
        '\tkw_foo [label="kw:foo"]\n'
        "\ta -> kw_foo [dir=both]\n"
        '\tb [label="b\n<some\n more content>" '
        'shape=note tooltip="<some\n more content>"]\n'
        '\thref_b [label="href:b"]\n'
        "\tb -> href_b [dir=back]\n"
        "\tb -> href_a [dir=forward]\n"
        "\tb -> kw_foo [dir=both]\n"
        '\tkw_bar [label="kw:bar"]\n'
        "\tb -> kw_bar [dir=both]\n"
        "}\n"
    )

    assert render_graphviz([doc1, doc2], engine="fdp").engine == "fdp"

    assert render_graphviz([doc1, doc2], node_colors={"a": "gold"}).source == (
        "digraph {\n"
        "\trankdir=LR\n"
        "\tnode [style=filled]\n"
        '\ta [label="a\nsome content" fillcolor=gold '
        'shape=note tooltip="some content"]\n'
        '\thref_a [label="href:a"]\n'
        "\ta -> href_a [dir=back]\n"
        '\tkw_foo [label="kw:foo"]\n'
        "\ta -> kw_foo [dir=both]\n"
        '\tb [label="b\n<some\n more content>" '
        'shape=note tooltip="<some\n more content>"]\n'
        '\thref_b [label="href:b"]\n'
        "\tb -> href_b [dir=back]\n"
        "\tb -> href_a [dir=forward]\n"
        "\tb -> kw_foo [dir=both]\n"
        '\tkw_bar [label="kw:bar"]\n'
        "\tb -> kw_bar [dir=both]\n"
        "}\n"
    )

    assert render_graphviz(
        [doc1, doc2], node_color="gold", node_colors={"a": None}
    ).source == (
        "digraph {\n"
        "\trankdir=LR\n"
        "\tnode [style=filled]\n"
        '\ta [label="a\nsome content" shape=note tooltip="some content"]\n'
        '\thref_a [label="href:a"]\n'
        "\ta -> href_a [dir=back]\n"
        '\tkw_foo [label="kw:foo"]\n'
        "\ta -> kw_foo [dir=both]\n"
        '\tb [label="b\n<some\n more content>" fillcolor=gold '
        'shape=note tooltip="<some\n more content>"]\n'
        '\thref_b [label="href:b"]\n'
        "\tb -> href_b [dir=back]\n"
        "\tb -> href_a [dir=forward]\n"
        "\tb -> kw_foo [dir=both]\n"
        '\tkw_bar [label="kw:bar"]\n'
        "\tb -> kw_bar [dir=both]\n"
        "}\n"
    )
