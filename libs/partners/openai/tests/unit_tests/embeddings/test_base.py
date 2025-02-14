import os
from unittest.mock import patch

import pytest

from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "foo"


def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbeddings(model_kwargs={"model": "foo"})


def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAIEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


def test_embed_documents_with_custom_chunk_size() -> None:
    embeddings = OpenAIEmbeddings(chunk_size=2, check_embedding_ctx_length=False)
    texts = ["text1", "text2", "text3", "text4"]
    custom_chunk_size = 3

    with patch.object(embeddings.client, "create") as mock_create:
        mock_create.side_effect = [
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
            {"data": [{"embedding": [0.5, 0.6]}, {"embedding": [0.7, 0.8]}]},
        ]

        result = embeddings.embed_documents(texts, chunk_size=custom_chunk_size)

        mock_create.call_args
        mock_create.assert_any_call(input=texts[0:3], **embeddings._invocation_params)
        mock_create.assert_any_call(input=texts[3:4], **embeddings._invocation_params)

    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
