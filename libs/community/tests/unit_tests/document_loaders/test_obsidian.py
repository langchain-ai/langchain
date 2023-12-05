from pathlib import Path

from langchain_community.document_loaders.obsidian import ObsidianLoader

OBSIDIAN_EXAMPLE_PATH = Path(__file__).parent / "sample_documents" / "obsidian"
STANDARD_METADATA_FIELDS = {
    "created",
    "path",
    "source",
    "last_accessed",
    "last_modified",
}

loader = ObsidianLoader(str(OBSIDIAN_EXAMPLE_PATH))
docs = loader.load()


def test_page_content_loaded() -> None:
    """Verify that all docs have page_content"""
    assert len(docs) == 6
    assert all(doc.page_content for doc in docs)


def test_disable_collect_metadata() -> None:
    """If collect_metadata is False, no additional metadata should be collected."""
    loader_without_metadata = ObsidianLoader(
        str(OBSIDIAN_EXAMPLE_PATH), collect_metadata=False
    )
    docs_wo = loader_without_metadata.load()
    assert len(docs_wo) == 6
    assert all(doc.page_content for doc in docs_wo)
    assert all(set(doc.metadata) == STANDARD_METADATA_FIELDS for doc in docs_wo)


def test_metadata_without_frontmatter() -> None:
    """Verify docs without frontmatter, still have basic metadata."""
    doc = next(doc for doc in docs if doc.metadata["source"] == "no_metadata.md")
    assert set(doc.metadata) == STANDARD_METADATA_FIELDS


def test_metadata_with_frontmatter() -> None:
    """Verify a standard frontmatter field is loaded."""
    doc = next(doc for doc in docs if doc.metadata["source"] == "frontmatter.md")
    assert set(doc.metadata) == STANDARD_METADATA_FIELDS | {"tags"}
    assert set(doc.metadata["tags"].split(",")) == {"journal/entry", "obsidian"}


def test_metadata_with_template_vars_in_frontmatter() -> None:
    """Verify frontmatter fields with template variables are loaded."""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "template_var_frontmatter.md"
    )
    FRONTMATTER_FIELDS = {
        "aString",
        "anArray",
        "aDict",
        "tags",
    }
    assert set(doc.metadata) == FRONTMATTER_FIELDS | STANDARD_METADATA_FIELDS
    assert doc.metadata["aString"] == "{{var}}"
    assert doc.metadata["anArray"] == "['element', '{{varElement}}']"
    assert doc.metadata["aDict"] == "{'dictId1': 'val', 'dictId2': '{{varVal}}'}"
    assert set(doc.metadata["tags"].split(",")) == {"tag", "{{varTag}}"}


def test_metadata_with_bad_frontmatter() -> None:
    """Verify a doc with non-yaml frontmatter."""
    doc = next(doc for doc in docs if doc.metadata["source"] == "bad_frontmatter.md")
    assert set(doc.metadata) == STANDARD_METADATA_FIELDS


def test_metadata_with_tags_and_frontmatter() -> None:
    """Verify a doc with frontmatter and tags/dataview tags are all added to
    metadata."""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "tags_and_frontmatter.md"
    )

    FRONTMATTER_FIELDS = {
        "aBool",
        "aFloat",
        "anInt",
        "anArray",
        "aString",
        "aDict",
        "tags",
    }
    DATAVIEW_FIELDS = {"dataview1", "dataview2", "dataview3"}
    assert (
        set(doc.metadata)
        == STANDARD_METADATA_FIELDS | FRONTMATTER_FIELDS | DATAVIEW_FIELDS
    )


def test_tags_in_page_content() -> None:
    """Verify a doc with tags are included in the metadata"""
    doc = next(doc for doc in docs if doc.metadata["source"] == "no_frontmatter.md")
    assert set(doc.metadata) == STANDARD_METADATA_FIELDS | {"tags"}


def test_boolean_metadata() -> None:
    """Verify boolean metadata is loaded correctly"""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "tags_and_frontmatter.md"
    )
    assert doc.metadata["aBool"]


def test_float_metadata() -> None:
    """Verify float metadata is loaded correctly"""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "tags_and_frontmatter.md"
    )
    assert doc.metadata["aFloat"] == 13.12345


def test_int_metadata() -> None:
    """Verify int metadata is loaded correctly"""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "tags_and_frontmatter.md"
    )
    assert doc.metadata["anInt"] == 15


def test_string_metadata() -> None:
    """Verify string metadata is loaded correctly"""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "tags_and_frontmatter.md"
    )
    assert doc.metadata["aString"] == "string value"


def test_array_metadata() -> None:
    """Verify array metadata is loaded as a string"""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "tags_and_frontmatter.md"
    )
    assert doc.metadata["anArray"] == "['one', 'two', 'three']"


def test_dict_metadata() -> None:
    """Verify dict metadata is stored as a string"""
    doc = next(
        doc for doc in docs if doc.metadata["source"] == "tags_and_frontmatter.md"
    )
    assert doc.metadata["aDict"] == "{'dictId1': '58417', 'dictId2': 1500}"
