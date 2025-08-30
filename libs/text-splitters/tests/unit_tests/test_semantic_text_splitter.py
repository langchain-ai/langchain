import pytest
from my_splitters.semantic_text_splitter import SemanticTextSplitter
from langchain.schema import Document
from typing import List

# Dummy embedding model for testing
class DummyEmbedder:
    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        # Simple embedding: vector of sentence length
        return [[len(d)] for d in docs]

def dummy_split(text: str) -> List[str]:
    return text.split(". ")

splitter = SemanticTextSplitter(
    embedding_model=DummyEmbedder(),
    sentence_splitter=dummy_split,
    mode="similarity",
    similarity_threshold=0.5,
    max_chunk_size=100,
)

# ---------- Tests ----------

def test_happy_path():
    text = (
        "Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. "
        "The sun was bright, and the air smelled of earth and fresh grass. "
        "The Indian Premier League (IPL) is the biggest cricket league in the world. "
        "People all over the world watch the matches and cheer for their favourite teams."
    )
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) >= 1

def test_split_documents():
    text1 = (
        "Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. "
        "The sun was bright, and the air smelled of earth and fresh grass."
    )
    text2 = (
        "Terrorism is a big danger to peace and safety. "
        "It causes harm to people and creates fear in cities and villages. "
        "When such attacks happen, they leave behind pain and sadness. "
        "To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety."
    )

    doc1 = Document(page_content=text1, metadata={"source": "text1"})
    doc2 = Document(page_content=text2, metadata={"source": "text2"})

    docs = splitter.split_documents([doc1, doc2])
    assert all(isinstance(d, Document) for d in docs)
    assert len(docs) >= 2
    assert all(d.page_content for d in docs)  # All chunks have content

def test_empty_text():
    assert splitter.split_text("") == []

def test_single_sentence():
    assert splitter.split_text("Single sentence.") == ["Single sentence."]
