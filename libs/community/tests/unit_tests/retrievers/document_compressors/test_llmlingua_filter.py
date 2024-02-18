import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from langchain_community.retrievers.document_compressors import LLMLinguaCompressor


# Mock PromptCompressor for testing purposes
class MockPromptCompressor:
    def compress_prompt(self, *args, **kwargs):
        """Mock behavior of the compress_prompt method"""
        return {
            "compressed_prompt": "<#ref0#> Compressed content for document 0 <#ref0#>\n\n"
            "<#ref1#> Compressed content for document 1 <#ref1#>"
        }


@pytest.fixture
def mock_prompt_compressor(mocker: MockerFixture):
    """Mock the external PromptCompressor dependency"""
    compressor = MockPromptCompressor()
    mocker.patch("llmlingua.PromptCompressor", return_value=compressor)
    return compressor


@pytest.fixture
def llm_lingua_compressor(mock_prompt_compressor):
    """Create an instance of LLMLinguaCompressor with the mocked PromptCompressor"""
    return LLMLinguaCompressor()


def test_format_context():
    """Test the _format_context method in the llmlinguacompressor"""
    docs = [
        Document(page_content="Content of document 0", metadata="1"),
        Document(page_content="Content of document 1", metadata="1"),
    ]
    formatted_context = LLMLinguaCompressor._format_context(docs)
    assert formatted_context == [
        "\n\n<#ref0#> Content of document 0 <#ref0#>\n\n",
        "\n\n<#ref1#> Content of document 1 <#ref1#>\n\n",
    ]


def test_extract_ref_id_tuples_and_clean(llm_lingua_compressor):
    """Test extracting reference ids from the documents contents"""
    contents = ["<#ref0#> Example content <#ref0#>", "Content with no ref ID."]
    result = llm_lingua_compressor.extract_ref_id_tuples_and_clean(contents)
    assert result == [("Example content", 0), ("Content with no ref ID.", -1)]


def test_compress_documents_no_documents(llm_lingua_compressor):
    """Test the compress_documents method with no documents"""
    result = llm_lingua_compressor.compress_documents([], "query")
    assert result == []


def test_compress_documents_with_documents(llm_lingua_compressor):
    """Test the compress_documents method with documents"""
    docs = [
        Document(content="Content of document 0", metadata="1"),
        Document(content="Content of document 1", metadata="2"),
    ]
    compressed_docs = llm_lingua_compressor.compress_documents(docs, "query")
    assert len(compressed_docs) == 2
    assert compressed_docs[0].page_content == "Compressed content for document 0"
    assert compressed_docs[0].metadata == "1"
    assert compressed_docs[1].page_content == "Compressed content for document 1"
    assert compressed_docs[1].metadata == "2"
