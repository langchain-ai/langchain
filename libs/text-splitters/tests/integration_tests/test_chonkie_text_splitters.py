"""Test ChonkieTextSplitter functionality."""
# ruff: noqa: PLC0415

import pytest
from langchain_core.documents import Document

# We don't import ChonkieTextSplitter at top level to avoid collection-time errors
# if chonkie is not installed, especially when we want to test ImportError.

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("chonkie", reason="chonkie not installed"),
    reason="chonkie not installed",
)


def test_chonkie_text_splitter_basic() -> None:
    """Test basic splitting with default chunker (recursive)."""
    from langchain_text_splitters.chonkie import ChonkieTextSplitter

    text = "This is a test. Here is another sentence."
    splitter = ChonkieTextSplitter()
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert any("test" in chunk for chunk in chunks)


def test_chonkie_text_splitter_with_alias() -> None:
    """Test splitting with a specific valid chunker alias."""
    from langchain_text_splitters.chonkie import ChonkieTextSplitter

    # Use the first available alias for testing
    aliases = ChonkieTextSplitter.valid_chunker_aliases
    assert "recursive" in aliases
    text = "Sentence one. Sentence two. Sentence three."
    splitter = ChonkieTextSplitter(chunker="recursive")
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chonkie_text_splitter_with_chunker_instance() -> None:
    """Test passing a chunker instance directly."""
    from chonkie.pipeline import ComponentRegistry  # type: ignore[import]

    from langchain_text_splitters.chonkie import ChonkieTextSplitter

    ChunkingClass = ComponentRegistry.get_chunker("recursive").component_class  # noqa: N806
    chunker_instance = ChunkingClass()
    text = "Direct instance test."
    splitter = ChonkieTextSplitter(chunker=chunker_instance)
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chonkie_text_splitter_invalid_alias() -> None:
    """Test that an invalid chunker alias raises an error."""
    from langchain_text_splitters.chonkie import ChonkieTextSplitter

    with pytest.raises(Exception):  # noqa: B017, PT011
        ChonkieTextSplitter(chunker="not_a_real_chunker")


def test_chonkie_text_splitter_metadata() -> None:
    """Test that metadata is preserved and enriched."""
    from langchain_text_splitters.chonkie import ChonkieTextSplitter

    text = "This is a long sentence. This is another one."
    splitter = ChonkieTextSplitter(chunker="sentence", chunk_size=20)
    docs = splitter.create_documents([text], metadatas=[{"source": "test"}])
    assert len(docs) > 0
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.metadata["source"] == "test"
        assert "start_index" in doc.metadata
        assert "end_index" in doc.metadata
        assert "token_count" in doc.metadata
        assert (
            doc.page_content
            == text[doc.metadata["start_index"] : doc.metadata["end_index"]]
        )


def test_chonkie_text_splitter_recursive_explicit() -> None:
    """Test recursive chunker explicitly."""
    from langchain_text_splitters.chonkie import ChonkieTextSplitter

    text = "Header\n\nSection 1\n\nSection 2"
    splitter = ChonkieTextSplitter(chunker="recursive", chunk_size=10)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    assert "Section 1" in text


def test_chonkie_text_splitter_overlap() -> None:
    """Test chunk overlap handling."""
    from langchain_text_splitters.chonkie import ChonkieTextSplitter

    text = "This is a somewhat long text to test the overlap features of the chunker."
    splitter = ChonkieTextSplitter(chunker="token", chunk_size=10, chunk_overlap=5)
    # The chunk_overlap should be passed to the underlying chunker
    assert splitter.chunker.chunk_overlap == 5
    chunks = splitter.split_text(text)
    assert len(chunks) > 1


def test_chonkie_text_splitter_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ImportError is raised if chonkie is not installed."""
    import sys
    from importlib import reload

    import langchain_text_splitters.chonkie as chonkie_mod

    # Store original modules to restore later
    original_modules = {
        "chonkie": sys.modules.get("chonkie"),
        "chonkie.chunker.base": sys.modules.get("chonkie.chunker.base"),
        "chonkie.pipeline": sys.modules.get("chonkie.pipeline"),
    }

    try:
        # Simulate chonkie not installed
        monkeypatch.setitem(sys.modules, "chonkie", None)
        monkeypatch.setitem(sys.modules, "chonkie.chunker.base", None)
        monkeypatch.setitem(sys.modules, "chonkie.pipeline", None)

        reload(chonkie_mod)
        with pytest.raises(ImportError):
            chonkie_mod.ChonkieTextSplitter()
    finally:
        # Restore original modules and reload to fix state for other tests
        for mod_name, mod_obj in original_modules.items():
            if mod_obj is None:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
            else:
                sys.modules[mod_name] = mod_obj
        reload(chonkie_mod)
