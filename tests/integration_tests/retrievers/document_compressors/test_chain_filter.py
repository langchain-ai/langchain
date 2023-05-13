"""Integration test for llm-based relevant doc filtering."""
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.schema import Document


def test_llm_chain_filter() -> None:
    texts = [
        "What happened to all of my cookies?",
        "I wish there were better Italian restaurants in my neighborhood.",
        "My favorite color is green",
    ]
    docs = [Document(page_content=t) for t in texts]
    relevant_filter = LLMChainFilter.from_llm(llm=ChatOpenAI())
    actual = relevant_filter.compress_documents(docs, "Things I said related to food")
    assert len(actual) == 2
    assert len(set(texts[:2]).intersection([d.page_content for d in actual])) == 2
