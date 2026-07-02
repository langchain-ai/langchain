import base64
from pathlib import Path

import pytest
from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_chroma.vectorstores import Chroma


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
    )


def test_similarity_search() -> None:
    """Test similarity search by Chroma."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1


def test_encode_image_rejects_path_traversal(tmp_path: Path) -> None:
    """Image URIs with ``..`` must be rejected."""
    store = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=10),
        image_root=tmp_path,
    )
    with pytest.raises(ValueError, match="Path traversal"):
        store.encode_image("../etc/passwd")


def test_encode_image_rejects_paths_outside_root(tmp_path: Path) -> None:
    """Resolved paths outside ``image_root`` must be rejected."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"png")

    store = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=10),
        image_root=image_root,
    )
    with pytest.raises(ValueError, match="outside allowed root"):
        store.encode_image(str(outside))


def test_encode_image_allows_paths_within_root(tmp_path: Path) -> None:
    """Valid image paths inside ``image_root`` should be readable."""
    image = tmp_path / "photo.png"
    image.write_bytes(b"test-image-data")

    store = Chroma(
        collection_name="test_collection",
        embedding_function=FakeEmbeddings(size=10),
        image_root=tmp_path,
    )
    assert store.encode_image("photo.png") == base64.b64encode(b"test-image-data").decode()
