import os

import pytest

from langchain.retrievers.document_compressors import CohereRerank

os.environ["COHERE_API_KEY"] = "foo"


@pytest.mark.requires("cohere")
def test_init() -> None:
    CohereRerank()

    CohereRerank(
        top_n=5, model="rerank-english_v2.0", cohere_api_key="foo", user_agent="bar"
    )
