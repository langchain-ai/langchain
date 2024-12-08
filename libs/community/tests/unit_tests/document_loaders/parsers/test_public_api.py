from langchain_community.document_loaders.parsers import __all__


def test_parsers_public_api_correct() -> None:
    """Test public API of parsers for breaking changes."""
    assert set(__all__) == {
        "AzureAIDocumentIntelligenceParser",
        "BS4HTMLParser",
        "DocAIParser",
        "GrobidParser",
        "LanguageParser",
        "OpenAIWhisperParser",
        "PyPDFParser",
        "PDFMinerParser",
        "PyMuPDFParser",
        "PyPDFium2TocParser",
        "PyPDFium2Parser",
        "PDFPlumberParser",
        "VsdxParser",
    }
