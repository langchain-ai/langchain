import pytest

from langchain import document_transformers

EXPECTED_DEPRECATED_IMPORTS = [
    "BeautifulSoupTransformer",
    "DoctranQATransformer",
    "DoctranTextTranslator",
    "DoctranPropertyExtractor",
    "EmbeddingsClusteringFilter",
    "EmbeddingsRedundantFilter",
    "GoogleTranslateTransformer",
    "get_stateful_documents",
    "LongContextReorder",
    "NucliaTextTransformer",
    "OpenAIMetadataTagger",
    "Html2TextTransformer",
]


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(document_transformers, import_)
            assert "langchain_community" in e
    with pytest.raises(AttributeError):
        getattr(document_transformers, "foo")
