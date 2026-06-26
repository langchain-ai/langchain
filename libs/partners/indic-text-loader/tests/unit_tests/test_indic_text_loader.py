"""Tests for IndicTextLoader."""
import pytest
import tempfile
import os
from langchain_community.document_loaders.indic_text_loader import (
    IndicTextLoader
)


@pytest.fixture
def hindi_text_file():
    """Create a temporary Hindi text file."""
    content = "यह एक परीक्षण दस्तावेज़ है। LangChain बहुत उपयोगी है।"
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.txt', 
        encoding='utf-8',
        delete=False
    ) as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)


@pytest.fixture  
def marathi_text_file():
    content = "हे एक चाचणी दस्तऐवज आहे। मराठी भाषा सुंदर आहे।"
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.txt',
        encoding='utf-8', 
        delete=False
    ) as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)


class TestIndicTextLoader:

    def test_load_hindi_document(self, hindi_text_file):
        loader = IndicTextLoader(
            file_path=hindi_text_file,
            language="hindi"
        )
        docs = loader.load()
        assert len(docs) == 1
        assert "परीक्षण" in docs[0].page_content
        assert docs[0].metadata["language"] == "hindi"
        assert docs[0].metadata["language_code"] == "hi"

    def test_load_marathi_document(self, marathi_text_file):
        loader = IndicTextLoader(
            file_path=marathi_text_file,
            language="marathi"
        )
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata["language_code"] == "mr"

    def test_unsupported_language_raises_error(self, hindi_text_file):
        with pytest.raises(ValueError, match="not supported"):
            IndicTextLoader(
                file_path=hindi_text_file,
                language="klingon"
            )

    def test_file_not_found_raises_error(self):
        loader = IndicTextLoader(
            file_path="nonexistent.txt",
            language="hindi"
        )
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_custom_metadata(self, hindi_text_file):
        loader = IndicTextLoader(
            file_path=hindi_text_file,
            language="hindi",
            metadata={"source_type": "government_doc"}
        )
        docs = loader.load()
        assert docs[0].metadata["source_type"] == "government_doc"

    def test_lazy_load(self, hindi_text_file):
        loader = IndicTextLoader(
            file_path=hindi_text_file,
            language="hindi"
        )
        docs = list(loader.lazy_load())
        assert len(docs) == 1

    def test_all_supported_languages(self, hindi_text_file):
        languages = [
            "hindi", "marathi", "bengali", "tamil",
            "telugu", "kannada", "malayalam", "gujarati",
            "punjabi", "odia", "urdu", "assamese"
        ]
        for lang in languages:
            loader = IndicTextLoader(
                file_path=hindi_text_file,
                language=lang
            )
            assert loader.language == lang
