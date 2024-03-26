"""Test ChatCohere chat model."""

from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.messages.human import HumanMessage


# def test_connectors() -> None:
#     """Test connectors parameter support from ChatCohere."""
#     llm = ChatCohere().bind(connectors=[{"id": "web-search"}])

#     result = llm.invoke("Who directed dune two? reply with just the name.")
#     assert isinstance(result.content, str)


def test_documents() -> None:
    """Test documents paraneter support from ChatCohere."""
    llm = ChatCohere()
    docs = [Document(page_content="The sky is green.")]

    prompt = HumanMessage("What color is the sky?", additional_kwargs={"documents": docs})

    result = llm.invoke([prompt])
    assert isinstance(result.content, str)
    assert len(result.response_metadata["documents"]) == 1
