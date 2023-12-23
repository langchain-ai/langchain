"""Test conversation chain and memory."""
from langchain_core.documents import Document
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough

from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
    create_conversational_retrieval_chain,
)
from langchain.llms.fake import FakeListLLM
from langchain.memory.buffer import ConversationBufferMemory
from tests.unit_tests.retrievers.sequential_retriever import SequentialRetriever


async def test_simplea() -> None:
    fixed_resp = "I don't know"
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(sequential_responses=[[]])
    memory = ConversationBufferMemory(
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
    memory = ConversationBufferMemory(
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
    memory = ConversationBufferMemory(
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
    memory = ConversationBufferMemory(
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


def test_create() -> None:
    answer = "I know the answer!"
    llm = FakeListLLM(responses=[answer])
    retriever = SequentialRetriever(
        sequential_responses=[[Document(page_content=answer)]]
    )
    question_gen_prompt = PromptTemplate.from_template("hi! {question} {chat_history}")
    combine_docs_chain = (
        PromptTemplate.from_template("combine! {input_documents}")
        | llm
        | {"answer": RunnablePassthrough()}
    )
    chain = create_conversational_retrieval_chain(
        llm, retriever, question_gen_prompt, combine_docs_chain
    )
    assert chain.invoke({"question": "What is the answer?", "chat_history": []}) == {
        "answer": "I know the answer!",
        "generated_question": "What is the answer?",
        "source_documents": [Document(page_content="I know the answer!")],
    }
