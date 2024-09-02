from langchain_core.graph_vectorstores.links import Link

from langchain_community.graph_vectorstores.extractors import HierarchyLinkExtractor

PATH_1 = ["Root", "H1", "h2"]

PATH_2 = ["Root", "H1"]

PATH_3 = ["Root"]


def test_up_only() -> None:
    extractor = HierarchyLinkExtractor()

    assert extractor.extract_one(PATH_1) == {
        # Path1 links up to Root/H1
        Link.outgoing(kind="hierarchy", tag="p:Root/H1"),
        # Path1 is linked to by stuff under Root/H1/h2
        Link.incoming(kind="hierarchy", tag="p:Root/H1/h2"),
    }

    assert extractor.extract_one(PATH_2) == {
        # Path2 links up to Root
        Link.outgoing(kind="hierarchy", tag="p:Root"),
        # Path2 is linked to by stuff under Root/H1/h2
        Link.incoming(kind="hierarchy", tag="p:Root/H1"),
    }

    assert extractor.extract_one(PATH_3) == {
        # Path3 is linked to by stuff under Root
        Link.incoming(kind="hierarchy", tag="p:Root"),
    }


def test_up_and_down() -> None:
    extractor = HierarchyLinkExtractor(child_links=True)

    assert extractor.extract_one(PATH_1) == {
        # Path1 links up to Root/H1
        Link.outgoing(kind="hierarchy", tag="p:Root/H1"),
        # Path1 is linked to by stuff under Root/H1/h2
        Link.incoming(kind="hierarchy", tag="p:Root/H1/h2"),
        # Path1 links down to things under Root/H1/h2.
        Link.outgoing(kind="hierarchy", tag="c:Root/H1/h2"),
        # Path1 is linked down to by Root/H1
        Link.incoming(kind="hierarchy", tag="c:Root/H1"),
    }

    assert extractor.extract_one(PATH_2) == {
        # Path2 links up to Root
        Link.outgoing(kind="hierarchy", tag="p:Root"),
        # Path2 is linked to by stuff under Root/H1/h2
        Link.incoming(kind="hierarchy", tag="p:Root/H1"),
        # Path2 links down to things under Root/H1.
        Link.outgoing(kind="hierarchy", tag="c:Root/H1"),
        # Path2 is linked down to by Root
        Link.incoming(kind="hierarchy", tag="c:Root"),
    }

    assert extractor.extract_one(PATH_3) == {
        # Path3 is linked to by stuff under Root
        Link.incoming(kind="hierarchy", tag="p:Root"),
        # Path3 links down to things under Root/H1.
        Link.outgoing(kind="hierarchy", tag="c:Root"),
    }


def test_sibling() -> None:
    extractor = HierarchyLinkExtractor(sibling_links=True, parent_links=False)

    assert extractor.extract_one(PATH_1) == {
        # Path1 links with anything else in Root/H1
        Link.bidir(kind="hierarchy", tag="s:Root/H1"),
    }

    assert extractor.extract_one(PATH_2) == {
        # Path2 links with anything else in Root
        Link.bidir(kind="hierarchy", tag="s:Root"),
    }

    assert extractor.extract_one(PATH_3) == {
        # Path3 links with anything else at the top level
        Link.bidir(kind="hierarchy", tag="s:"),
    }
