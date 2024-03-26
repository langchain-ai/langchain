"""Test ChatCohere chat model."""

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSerializable,
)

from langchain_cohere import ChatCohere


def test_connectors() -> None:
    """Test connectors parameter support from ChatCohere."""
    llm = ChatCohere().bind(connectors=[{"id": "web-search"}])

    result = llm.invoke("Who directed dune two? reply with just the name.")
    assert isinstance(result.content, str)


def test_documents() -> None:
    """Test documents paraneter support from ChatCohere."""
    docs = [{"text": "The sky is green."}]
    llm = ChatCohere().bind(documents=docs)
    prompt = "What color is the sky?"

    result = llm.invoke(prompt)
    assert isinstance(result.content, str)
    assert len(result.response_metadata["documents"]) == 1


def test_documents_chain() -> None:
    """Test documents paraneter support from ChatCohere."""
    llm = ChatCohere()

    def get_documents(_: Any) -> List[Document]:
        return [Document(page_content="The sky is green.")]

    def format_input_msgs(input: Dict[str, Any]) -> List[HumanMessage]:
        return [
            HumanMessage(
                content=input["message"],
                additional_kwargs={
                    "documents": input.get("documents", None),
                },
            )
        ]

    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("input_msgs")])
    chain: RunnableSerializable[Any, Any] = (
        {"message": RunnablePassthrough(), "documents": get_documents}
        | RunnablePassthrough()
        | {"input_msgs": format_input_msgs}
        | prompt
        | llm
    )

    result = chain.invoke("What color is the sky?")
    assert isinstance(result.content, str)
    assert len(result.response_metadata["documents"]) == 1
