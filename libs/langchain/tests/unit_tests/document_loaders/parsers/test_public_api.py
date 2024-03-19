import pytest

from langchain.document_loaders import parsers


def test_deprecated_error() -> None:
    """Hard-code public API to help determine if we have broken it."""
    deprecated = [
        "BS4HTMLParser",
        "DocAIParser",
        "GrobidParser",
        "LanguageParser",
        "OpenAIWhisperParser",
        "PyPDFParser",
        "PDFMinerParser",
        "PyMuPDFParser",
        "PyPDFium2Parser",
        "PDFPlumberParser",
    ]
    for import_ in deprecated:
        with pytest.raises(ImportError) as e:
            getattr(parsers, import_)
            assert "langchain_community" in e.msg
