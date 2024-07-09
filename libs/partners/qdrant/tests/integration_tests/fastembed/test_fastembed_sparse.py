import numpy as np
import pytest

from langchain_qdrant import FastEmbedSparse


@pytest.mark.parametrize(
    "model_name", ["Qdrant/bm25", "Qdrant/bm42-all-minilm-l6-v2-attentions"]
)
def test_attention_embeddings(model_name: str) -> None:
    model = FastEmbedSparse(model_name=model_name)

    query_output = model.embed_query("Stay, steady and sprint.")

    assert len(query_output.indices) == len(query_output.values)
    assert np.allclose(query_output.values, np.ones(len(query_output.values)))

    texts = [
        "The journey of a thousand miles begins with a single step.",
        "Be yourself in a world that is constantly trying to make you something else",
        "In the end, we only regret the chances we didn't take.",
        "Every moment is a fresh beginning.",
        "Not all those who wander are lost.",
        "Do not go where the path may lead, go elsewhere and leave a trail.",
        "Life is what happens when you're busy making other plans.",
        "The only limit to our realization of tomorrow is our doubts of today.",
    ]

    output = model.embed_documents(texts)

    assert len(output) == len(texts)

    for result in output:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) > 0
