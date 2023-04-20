"""Integration test for LLMChainExtractor."""
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document


def test_llm_chain_extractor() -> None:
    texts = [
        "The Roman Empire followed the Roman Republic.",
        "I love chocolate chip cookies—my mother makes great cookies.",
        "The first Roman emperor was Caesar Augustus.",
        "Don't you just love Caesar salad?",
        "The Roman Empire collapsed in 476 AD after the fall of Rome.",
        "Let's go to Olive Garden!",
    ]
    doc = Document(page_content=" ".join(texts))
    compressor = LLMChainExtractor.from_llm(ChatOpenAI())
    actual = compressor.compress_documents([doc], "Tell me about the Roman Empire")[
        0
    ].page_content
    expected_returned = [0, 2, 4]
    expected_not_returned = [1, 3, 5]
    assert all([texts[i] in actual for i in expected_returned])
    assert all([texts[i] not in actual for i in expected_not_returned])


def test_llm_chain_extractor_empty() -> None:
    texts = [
        "I love chocolate chip cookies—my mother makes great cookies.",
        "Don't you just love Caesar salad?",
        "Let's go to Olive Garden!",
    ]
    doc = Document(page_content=" ".join(texts))
    compressor = LLMChainExtractor.from_llm(ChatOpenAI())
    actual = compressor.compress_documents([doc], "Tell me about the Roman Empire")
    assert len(actual) == 0
