"""Unit tests for metadata hydration in TextSplitter."""
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


def test_metadata_hydration_adds_index() -> None:
    """Test that the hydrator can add a simple chunk index."""

    # 1. Define the logic: Add index to metadata
    def add_index_logic(doc: Document, index: int) -> Dict[str, Any]:
        return {"chunk_index": index}

    # 2. Setup Splitter with the hook
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1,
        chunk_overlap=0,
        metadata_hydrator=add_index_logic
    )

    # 3. Create documents
    text = "One Two Three"
    docs = splitter.create_documents([text])

    # 4. Assertions (Check results)
    assert len(docs) == 3
    assert docs[0].page_content == "One"
    assert docs[0].metadata["chunk_index"] == 0

    assert docs[1].page_content == "Two"
    assert docs[1].metadata["chunk_index"] == 1

    assert docs[2].page_content == "Three"
    assert docs[2].metadata["chunk_index"] == 2


def test_metadata_hydration_preserves_existing_metadata() -> None:
    """Test that existing metadata passed to create_documents is not lost."""

    def add_version(doc: Document, index: int) -> Dict[str, Any]:
        return {"version": "v1.0", "chunk_id": index}

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1,
        chunk_overlap=0,
        metadata_hydrator=add_version
    )

    text = "Alpha Beta"
    # User passes existing metadata (e.g., source file name)
    original_metadata = [{"source": "file.txt", "author": "Kamran"}]

    docs = splitter.create_documents([text], metadatas=original_metadata)

    assert len(docs) == 2

    # Check First Chunk
    # "source" and "author" should come from original_metadata
    # "version" and "chunk_id" should come from hydration
    assert docs[0].metadata["source"] == "file.txt"
    assert docs[0].metadata["author"] == "Kamran"
    assert docs[0].metadata["version"] == "v1.0"
    assert docs[0].metadata["chunk_id"] == 0


def test_metadata_hydration_conditional_logic() -> None:
    """Test that hydrator can use document content to make decisions."""

    def tag_important(doc: Document, index: int) -> Dict[str, Any]:
        # Logic: If content is "Urgent", add a priority tag
        if "Urgent" in doc.page_content:
            return {"priority": "high"}
        return {"priority": "normal"}

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=20,
        chunk_overlap=0,
        metadata_hydrator=tag_important
    )

    texts = ["This is Normal.", "This is Urgent!"]
    docs = splitter.create_documents(texts)

    # Check logic
    assert docs[0].metadata["priority"] == "normal"
    assert docs[1].metadata["priority"] == "high"
