import pytest

from langchain.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank


@pytest.mark.requires("langchain_openai")
def test__list_rerank_init() -> None:
    from langchain_openai import ChatOpenAI

    LLMListwiseRerank.from_llm(
        llm=ChatOpenAI(api_key="foo"),  # type: ignore[arg-type]
        top_n=10,
    )
