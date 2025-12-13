import pytest

from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)

try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers is not installed",
)
def test_sentence_transformers_encode_returns_token_ids() -> None:
    splitter = SentenceTransformersTokenTextSplitter()
    tokens = splitter.encode(text="Hello world")

    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)
