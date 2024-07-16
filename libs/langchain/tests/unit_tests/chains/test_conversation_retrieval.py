"""Test conversation chain and memory."""

from langchain_core.documents import Document
from langchain_core.language_models import FakeListLLM

from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
)
from langchain.memory.buffer import ConversationBufferMemory
from tests.unit_tests.retrievers.sequential_retriever import SequentialRetriever


async def test_simplea() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(sequential_responses=[[]])
    memory = ConversationBufferMemory(  # type: ignore[call-arg]
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = await qa_chain.acall("What is the answer?")
    assert got["chat_history"][1].content == fixed_resp
    assert got["answer"] == fixed_resp


async def test_fixed_message_response_when_docs_founda() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(
        sequential_responses=[[Document(page_content=answer)]]
    )
    memory = ConversationBufferMemory(  # type: ignore[call-arg]
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = await qa_chain.acall("What is the answer?")
    assert got["chat_history"][1].content == answer
    assert got["answer"] == answer


def test_fixed_message_response_when_no_docs_found() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(sequential_responses=[[]])
    memory = ConversationBufferMemory(  # type: ignore[call-arg]
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = qa_chain("What is the answer?")
    assert got["chat_history"][1].content == fixed_resp
    assert got["answer"] == fixed_resp


def test_fixed_message_response_when_docs_found() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(
        sequential_responses=[[Document(page_content=answer)]]
    )
    memory = ConversationBufferMemory(  # type: ignore[call-arg]
        k=1, output_key="answer", memory_key="chat_history", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        rephrase_question=False,
        response_if_no_docs_found=fixed_resp,
        verbose=True,
    )
    got = qa_chain("What is the answer?")
    assert got["chat_history"][1].content == answer
    assert got["answer"] == answer
